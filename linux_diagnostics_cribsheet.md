# Linux Performance Diagnostics: Crib Sheet

Commands used while profiling and debugging the BPE trainer, with notes on how to read them.

---

## 1. Time and peak memory of a whole run

```sh
/usr/bin/time -v uv run python driver.py
```

- Use the full path `/usr/bin/time`. Plain `time` is a **bash keyword** that has no `-v` and only prints real/user/sys.
- The report goes to stderr after your program's own output.

| Field | Meaning |
|---|---|
| `Elapsed (wall clock)` | Real time |
| `User time` | CPU time running your code |
| `System time` | CPU time inside the kernel (syscalls, page faults). If this is high, investigate. |
| `Percent of CPU` | ~100% = one core; >100% = parallel |
| `Maximum resident set size` | Peak RAM (KB) of the **largest single process**, not the sum across worker processes |
| `Major page faults` | Faults that needed disk I/O (swap or file). Should be ~0. |
| `Minor page faults` | Pages touched for the first time; roughly tracks memory allocated |
| `Involuntary context switches` | Times the OS preempted the process; high when processes compete for cores |
| `File system inputs` | Blocks read from disk (0 = file was already in the page cache) |

**Measure a baseline first:** run the same script with the expensive call commented out to separate startup/import cost from real work. (Example: an indirect `import torch` cost ~560 MB and ~2 s.)

---

## 2. Memory pressure and swapping

### System-wide overview
```sh
free -h                 # total / used / available RAM and swap
vmstat 1                # one line per second; Ctrl+C to stop
```

`vmstat` columns to watch:

| Column | Meaning | Worry when… |
|---|---|---|
| `si` | Swap **in** (KB/s) | Large and sustained in both directions |
| `so` | Swap **out** (KB/s) | **Steadily non-zero → RAM is exhausted right now** |
| `free` | Free RAM (KB) | Near zero *and* `so` > 0 |
| `wa` | % CPU waiting on I/O | High along with swap activity |
| `us` / `sy` | % CPU in user / kernel | `sy` unexpectedly high → syscall-heavy code |

- **Swap that's in use but not moving (`so` = 0) is harmless.** Pages from earlier memory pressure stay in swap until something needs them, and a small `si` trickle is just those pages being read back.

### Per-process memory
```sh
htop
```
- **F4** filter (e.g. `python`) · **F5** tree view (main process + workers) · **F6** sort by `M_RESIDENT` / `PERCENT_MEM`
- **`RES`** = resident RAM: the number to watch. Ignore `VIRT` (reserved address space).
- **Summing `RES` over-counts** forked workers, because shared pages are counted in each process. For an accurate total, watch the **`Mem` bar** before/during the run, or sum PSS:

```sh
grep Pss /proc/<PID>/smaps_rollup          # proportional share of shared pages
grep -E "VmRSS|VmSwap" /proc/<PID>/status  # resident and swapped-out amounts for one process
```

### Is a process thrashing or actually computing?
```sh
awk '{print "majflt",$12,"utime",$14,"stime",$15}' /proc/<PID>/stat; sleep 5; \
awk '{print "majflt",$12,"utime",$14,"stime",$15}' /proc/<PID>/stat
```
- `utime`/`stime` are in clock ticks (usually 100/s). A rise of ~500 in `utime` over 5 s = 100% of a core doing real work.
- `majflt` rising = waiting on disk/swap (thrashing).

---

## 3. Finding processes (including orphans)

```sh
pgrep -af driver.py                                   # PIDs + full command lines
ps -eo pid,ppid,etime,stat,%cpu,rss,cmd | grep "[d]river.py"
```
- `etime` = how long the process has been running; `stat` `R` = running, `S` = sleeping.
- `ppid` of 1 or a systemd process → **orphan** (its parent died). Interrupted `ProcessPoolExecutor` runs can leave workers behind, each still holding memory.
- `[d]river.py` stops `grep` from matching its own command line.

```sh
kill <PID> ...          # polite (SIGTERM)
kill -9 <PID> ...       # forceful, if it won't exit
```

---

## 4. Where is the CPU going? (Python hot spots)

### cProfile: per-function totals (re-runs the program)
```sh
uv run python -m cProfile -s cumtime script.py > profile.txt
```
- `tottime` = time in the function's own lines; `cumtime` = including functions it calls.
- Adds overhead to **every call**, so it inflates code made of many tiny calls. Use it to **rank** hot spots, not as true durations or to compare runs. For real durations, put `time.perf_counter()` around each phase.
- Import time (e.g. `torch`) shows up at the top of `cumtime`; skip down to your own functions.

### py-spy: sample a **running** process without stopping it
```sh
sudo "$(which py-spy)" dump --pid <PID>                    # current stack, right now
sudo "$(which py-spy)" top  --pid <PID>                    # live top-like view by function
sudo "$(which py-spy)" record --pid <PID> --duration 60 -o profile.svg   # flame graph with line numbers
```
- Run `dump` several times a few seconds apart; if it's always on the same line, that's the hot spot.
- `top`: **`%Own`** = the function's own lines; **`%Total`** = including callees.
- Flame graph: open the `.svg` in a browser; block width = share of time.
- Line attribution can be slightly off on Python 3.12+. If a result looks implausible, confirm it with an experiment.

**Why `sudo "$(which py-spy)"`?**
- Without sudo: Ubuntu's Yama setting (`kernel.yama.ptrace_scope=1`) only allows attaching to your own *child* processes → "Permission Denied".
- With plain `sudo py-spy`: `sudo` resets `PATH`, so a conda/venv-installed py-spy isn't found.
- `$(which py-spy)` is expanded by *your* shell first, so root runs exactly that binary.
- Optional, until reboot: `sudo sysctl kernel.yama.ptrace_scope=0` lets same-user tools attach without sudo (weaker security).

**Hot-loop gotcha:** `logger.debug(f"...{big_obj}...")` builds the f-string **before** the level check, even when DEBUG is off. Use lazy `%s` arguments, or check the level first. (See the Python logging HOWTO, "Optimization".)

---

## 5. Where is the *system* time going? (syscalls)

### Summary table
```sh
strace -c -f -o strace.txt .venv/bin/python driver.py
```
- `-c` = summary per syscall (count, time); `-f` = follow child processes/threads; `-o` = write to a file.
- Run the venv's `python` directly (not via `uv run`) so `uv`'s own syscalls don't clutter the results.
- Ignore `futex` / `wait4`: that's blocked/waiting time, not CPU.
- strace slows down every syscall, so use it to see **which** calls dominate, not how long they normally take.

### Who is making a syscall? (stack traces)
```sh
strace -f -k -e trace=clock_gettime -o strace_k.txt .venv/bin/python driver.py
```
- `-k` prints a **stack trace per call**, so use a **small input** (`corpus.en`) or the output gets huge.
- Summarize the callers:
```sh
grep -c 'clock_gettime(' strace_k.txt                                   # number of calls
grep -o '/[^ (]*\.so[^ (]*' strace_k.txt | sort | uniq -c | sort -rn | head   # shared libs in the stacks
grep '_regex' strace_k.txt | sed 's/.*\.so//' | sort | uniq -c | sort -rn | head   # functions within one lib
```
- Statically linked Python builds show interpreter frames as `.../bin/python3.13`, not `libpython.so`.

---

## 6. Clock source (why `clock_gettime` became a real syscall)

```sh
cat /sys/devices/system/clocksource/clocksource0/current_clocksource
cat /sys/devices/system/clocksource/clocksource0/available_clocksource
sudo dmesg | grep -i -E "tsc|clocksource"
cat /proc/cmdline                     # kernel boot parameters for this boot
```
- `tsc` = fast clock reads through the vDSO, no kernel entry. `hpet` / `acpi_pm` = every clock read is a syscall (~1.4 µs on this machine).
- On this machine the kernel drops TSC at boot because of a 0.17% calibration mismatch, and a BIOS update didn't fix it. Code that reads the clock per operation (e.g. `regex.finditer` per match) therefore shows huge `sys` time. `findall` avoided it.

### Is the system clock accurate? (NTP)
```sh
timedatectl timesync-status           # systemd-timesyncd: look at Frequency and Offset
chronyc tracking                      # if running chrony instead
```
- `Frequency` of a few ppm = healthy. Pinned at ±500 ppm with a growing `Offset` = the clock is running at the wrong rate.

### Hardware/firmware identity
```sh
cat /sys/class/dmi/id/{board_vendor,board_name,bios_version,bios_date}
grep -m1 "model name" /proc/cpuinfo
```

### Trying a kernel parameter for a single boot (no permanent change)
At the GRUB menu press `e`, append parameters to the `linux` line, then press `Ctrl+X` to boot. A normal reboot reverts it. To make a change permanent, edit `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then run `sudo update-grub`.

---

## 7. Quick checklist

| Symptom | First command |
|---|---|
| "How long / how much RAM?" | `/usr/bin/time -v …` |
| Machine sluggish, memory high | `free -h`, `vmstat 1` (`so`), htop sorted by `RES` |
| Leftover processes? | `ps -eo pid,ppid,etime,stat,rss,cmd \| grep "[d]river.py"` |
| Which Python function is slow? | `py-spy top` / `record` (live) or cProfile (re-run) |
| Long-running job looks stuck | `py-spy dump` a few times |
| `sys` time unexpectedly high | `strace -c -f`, then `strace -k` on a small input |
| Clock syscalls dominate | check `current_clocksource` |
| Parallel speedup lower than expected | compare `Percent of CPU`, involuntary context switches, and per-phase `perf_counter` timings |

from tests.test_train_bpe import test_train_bpe

import time

def walltime():
    times = []
    for i in range(3):
        # 1. Start the timer
        start_time = time.perf_counter()
        # 2. Run your code block
        test_train_bpe()
        # 3. Stop the timer
        end_time = time.perf_counter()
        # 4. Calculate the duration
        execution_time = end_time - start_time
        times.append(execution_time)
    for t in times:
        print(f"Code took {t:.6f} seconds to run.")
        
    
    
walltime()

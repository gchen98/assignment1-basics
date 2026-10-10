import math
import torch
from einops import rearrange,einsum,repeat



import logging

# from cs336_basics.basic_building_blocks import in_embeddings

# from cs336_basics.basic_building_blocks import d_model, d_ff, max_seq_len

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a console handler
console_handler = logging.StreamHandler()
# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)

class Linear(torch.nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        sigma = math.sqrt(2/(in_features+out_features))
        # weight =
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(out_features,in_features),-3*sigma,3*sigma))
        # print(f"weights are {self.weight}")
        self.device = device
        self.dtype = dtype

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return einsum(self.weight,x,"... d_out d_in,  ... d_in-> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        self.embedding_matrix=torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(num_embeddings,embedding_dim),0,1))
        self.device = device
        self.dtype = dtype

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.embedding_matrix[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self,d_model:int, eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.gain =  torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape.
        # d_model = 10
        # epsilon = 1e-5
        # x = torch.rand(10)
        from einops import reduce
        in_dtype = x.dtype
        print(f"Input tensor is of shape {x.shape} model dim is {self.d_model}")
        x2 = x ** 2
        ssq = reduce(x2, '... last -> ... 1', 'sum')
        rms = torch.sqrt(ssq / self.d_model + self.eps)
        rms_norm = (x * self.gain) / rms
        out = rms_norm.to(in_dtype)
        return out

class FFN(torch.nn.Module):
    def __init__(self,d_model:int, d_ff:int|None, device=None,dtype=None):
         super().__init__()
         self.d_model = d_model
         if d_ff is None:
            self.d_ff = round(d_model/64)*64
         else:
            self.d_ff = d_ff
         # print(f"Model size {self.d_model} and d_ff {self.d_ff}")
         self.w1 = Linear(self.d_model,self.d_ff)
         # print(f"weights for self.w1 is {self.w1.weight.shape}")
         self.w3 = Linear(self.d_model,self.d_ff)
         self.w2 = Linear(self.d_ff,self.d_model)

    def forward(self,in_features):
        w1_x = self.w1(in_features)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(in_features)
        silu_w1_x_w3_x = silu_w1_x * w3_x
        swiglu = self.w2(silu_w1_x_w3_x)
        return swiglu

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta = theta # theta value for the RoPE
        self.d_k = d_k # dimension of query and key vectors
        self.max_seq_len = max_seq_len # maximum sequence length that will be input
        # create a tensor that stores for each pair of positions along the embedding dimension
        # their appropriate rotation angles where each element
        # is 1.0 / theta ^ (0/embeding_dim), 1.0 / theta ^ ((2)/embeding_dim),...
        angles = 1.0 / 10000 ** (torch.arange(0, d_k, 2, dtype=torch.float) / d_k)
        logger.debug("angles %s of size %d ", angles[0:5], len(angles))
        # create a tensor of sequence positions [0..max_seq_len]
        token_positions = torch.arange(max_seq_len)
        # take the outer product so that each row of the output matrix is the angles vector scaled by each element of the position index. Lower coefficients (position index) means the angles are closer to zero.
        scaled_angles = torch.einsum("p,f->pf", token_positions, angles)
        logger.debug("Scaled inv freq %s", scaled_angles[:4, :4])
        # because we are working with pairs where each pair shares the same angle, duplicate these angles
        # and output a square matrix that has dimensions sequence length
        angles =  repeat(scaled_angles, "... f-> ... (f r)", r=2)
        self.register_buffer("angles",angles,persistent=False)



    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        def rotate_half(query_tokens: torch.Tensor) -> torch.Tensor:
            logger.debug("Input tensor shape %s", query_tokens.shape)
            # take the last dimension and split it into chunks of 2
            query_tokens = rearrange(query_tokens, ' ... (pos emb) -> ... pos emb', emb=2)
            logger.debug("Input tensor after splitting last dimension is shape %s", query_tokens.shape)
            # create two tensor slices, each slice containing matching element across the two chunks created above.
            # one slice contains odd elements the other even elements assuming 1-index. We lose a dimension.
            u1, u2 = query_tokens.unbind(dim=-1)
            logger.debug("Odd (1,3,5,...) slice is shape %s", u1.shape)
            logger.debug("Even (2,4,6,...) slice is shape %s", u2.shape)
            # print(f"u1: {u1[0]}\nu2: {u2[0]}")
            # Along the last dimension of interesting, stack the second slice to the first slice element wise, negating the even slice
            # We gain a dimension
            u3 = torch.stack((-u2, u1), dim=-1)
            logger.debug("Stacked slices is now shape %s", u3.shape)
            # print(f"u3 {u3[0]}")
            # collapse the last two dimensions into one
            merged_u3 = rearrange(u3, '... d r -> ... (d r)')
            logger.debug("Merged last two dimensions. Matrix is now dimension %s", merged_u3.shape)
            # print(f"merged_u3 {merged_u3}")
            return merged_u3

        def do_rotate(query_tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
            # the second to the last dimension is the sequence length
            num_tokens = query_tokens.shape[-2]
            # we are concerned only with up to the sequence length in our datastructure
            pos_enc = angles[:num_tokens]
            # q*pos_enc.cos()
            rotation = query_tokens * pos_enc.cos() + (rotate_half(query_tokens) * pos_enc.sin())
            return rotation


        rotation = do_rotate(x, self.angles)
        logger.debug("Rotation matrix is %s", rotation.shape)
        return rotation

        # x is of shape (...,seq_len, d_k). return a tensor of the same shape




class Softmax(torch.nn.Module):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    def __init__(self):
        super().__init__()

    def forward(self,in_features:torch.Tensor,dim:int)->torch.Tensor:
        # print(f"x is {x} with shape {x.shape}")
        # dim = -1
        max = torch.max(in_features, dim, keepdim=True).values
        # print(f"max is {max} with shape {max.shape}")
        x_smaller = torch.subtract(in_features, max)
        # print(f"x_smaller {x_smaller} with shape {x_smaller.shape}")
        x_exp = torch.exp(x_smaller)
        # print(f"x_exp {x_exp} ")
        denom = torch.sum(x_exp, dim=-1, keepdim=True)
        # print(f"denom {denom}")
        softmax = x_exp / denom
        # print(f"softmax {softmax}")
        return softmax

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        embedding_dim = Q.shape[-1]
        from cs336_basics.custom_modules import Softmax
        softmax = Softmax()
        denom =  math.sqrt(embedding_dim)
        qk = einsum(Q, K, "... n d_k , ... m d_k -> ... n m ") / denom
        # print(f"qk is {qk.shape} {qk}")
        subtractor = lambda val: val - torch.inf
        qk_masked = torch.where(~mask, subtractor(qk), qk)
        # print(f"qk2 is {qk_masked.shape} {qk_masked}")
        softmaxed = softmax(qk_masked, dim=-1)
        # print(f"softmaxed = {softmaxed.shape}")
        attn = einsum(softmaxed, V,
                      "... n d_v ,... d_v m    ->... n m   ")
        # print(f"attn {attn}")
        return attn

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self,rope:RotaryPositionalEmbedding, max_seq_len:int,d_model:int,num_heads:int,
                 q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor,
                 v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor):
        super().__init__()
        self.rope = rope
        #self attention so these are the same
        n = max_seq_len
        m = max_seq_len
        d_k = d_model
        d_v = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.q_proj = Linear(num_heads * d_k,d_model)
        self.q_proj.weight = torch.nn.Parameter(q_proj_weight)
        self.k_proj = Linear(num_heads * d_k, d_model)
        self.k_proj.weight = torch.nn.Parameter(k_proj_weight)
        self.v_proj = Linear(num_heads * d_v, d_model)
        self.v_proj.weight = torch.nn.Parameter(v_proj_weight)
        self.output_proj = Linear(d_model, num_heads * d_model)
        self.output_proj.weight = torch.nn.Parameter(o_proj_weight)
        self.sdpa = ScaledDotProductAttention()



    def forward(self,in_features:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        """
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on. )
        """
        # d_k = q_proj_weight.shape[-2]/num_heads
        # d_v = v_proj_weight.shape[-2]/num_heads

        # self.q_proj.weight = torch.nn.Parameter(q_proj_weight)
        # logger.debug(f"q weight {q_proj_weight.shape} in features {in_features.shape}")
        q_projection = self.q_proj(in_features)
        # logger.debug(f"q project {q_projection.shape}")
        q_slices = rearrange(q_projection, '... seq_len (h d_w) -> ... h seq_len d_w ', h=self.num_heads)
        logger.debug(f"q slices {q_slices.shape}")


        k_projection = self.k_proj(in_features)
        k_slices = rearrange(k_projection, '... seq_len (h d_w) -> ... h seq_len d_w ', h=self.num_heads)

        # q_slices = get_sliced(q_proj_weight, in_features)
        # k_slices = get_sliced(k_proj_weight, in_features)
        if token_positions is not None:
            q_slices = self.rope(q_slices,token_positions)
            k_slices = self.rope(k_slices,token_positions)


        v_projection = self.v_proj(in_features)
        v_slices = rearrange(v_projection, '... seq_len (h d_w) -> ... h seq_len d_w ', h=self.num_heads)
        # v_slices = get_sliced(v_proj_weight, in_features)

        # from cs336_basics.custom_modules import ScaledDotProductAttention
        # sdpa = ScaledDotProductAttention()
        mask = ~torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool()
        multihead = self.sdpa(q_slices, k_slices, v_slices, mask)
        # print(f"multihead {multihead.shape}")
        merged_multihead = rearrange(multihead, "... heads seq_len d_w -> ... seq_len (heads d_w)", heads=self.num_heads)
        logger.debug(f"merged multihead {merged_multihead.shape}")
        # o_project = einsum(o_proj_weight, merged_multihead, "emb_dim h_d_v , ... h_d_v -> ... emb_dim ")

        o_projection = self.output_proj(merged_multihead)
        # logger.debug(f"o project {o_projection.shape}")
        # print(f"o_project shape {o_project.shape}")
        # print(f"mask has nan? {torch.isnan(multihead).any()}")
        return o_projection
        return o_project

class TransformerBlock(torch.nn.Module):
    def __init__(self,d_model,num_heads, d_ff, max_seq_len, theta, in_features):
                 #
                 # attn.q_proj.weight,attn.k_proj.weight):
        super().__init__()
        rope = RotaryPositionalEmbedding(theta,d_model,max_seq_len,None)
        self.attn = MultiheadSelfAttention(rope)

    # def __init__(self,d_model,num_heads, d_ff, max_seq_len, theta, in_features,attn.q_proj.weight,attn.k_proj.weight,attn.v_proj.weight,ffn.w2.weight,ffn.w3.weight,ln1.weight,ln2.weight):


        # super().__init__()
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_ff = d_ff
        # self.max_seq_len = max_seq_len



        # d_model 64 num_heads 4 d_ff 128 max_seq_len 16 theta 10000.0
        # in features shape torch.Size([4, 12, 64])

        # weight attn.q_proj.weight has shape torch.Size([64, 64])
        # weight attn.k_proj.weight has shape torch.Size([64, 64])
        # weight attn.v_proj.weight has shape torch.Size([64, 64])
        # weight attn.output_proj.weight has shape torch.Size([64, 64])
        # weight ffn.w1.weight has shape torch.Size([128, 64])
        # weight ffn.w2.weight has shape torch.Size([64, 128])
        # weight ffn.w3.weight has shape torch.Size([128, 64])
        # weight ln1.weight has shape torch.Size([64])
        # weight ln2.weight has shape torch.Size([64])


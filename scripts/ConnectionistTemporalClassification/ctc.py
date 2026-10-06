import torch

NEG_INF = torch.finfo(torch.float).min

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="none"):
    seq_len, batch_size, num_classes = log_probs.shape
    B = torch.arange(batch_size)

    _t_a_r_g_e_t_s_ = torch.cat([targets, torch.zeros(batch_size, 1, device=log_probs.device)])

    print(_t_a_r_g_e_t_s_)

if __name__ == "__main__":
    T,B,C = 128, 2, 32
    t = 50
    blank = 0
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    logits = torch.randn(T,B,C).requires_grad_().to(device)
    log_probs = logits.log_softmax(dim=-1).to(device)
    targets = torch.randint(1,C,(B,t),dtype=torch.long).to(device)
    input_lengths = torch.full((B,),T,dtype=torch.long).to(device)
    target_lengths = torch.full((B,), t,dtype=torch.long).to(device)
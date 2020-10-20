torch.cuda.device_count()可以获得能够使用的GPU数量



unset all_proxy && unset ALL_PROXY

torch.Tensor

1. CPU tensor转GPU tensor：

cpu_imgs.cuda()
2. GPU tensor 转CPU tensor：

gpu_imgs.cpu()
3. numpy转为CPU tensor：

torch.from_numpy( imgs )
4.CPU tensor转为numpy数据：

cpu_imgs.numpy()
5. note：GPU tensor不能直接转为numpy数组，必须先转到CPU tensor。

6. 如果tensor是标量的话，可以直接使用 item() 函数（只能是标量）将值取出来：

print loss_output.item()
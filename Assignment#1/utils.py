# make log.txt-1680854493 pretty

def one_set(lines):
    arch = lines[0][0]
    num_block = lines[0][1]
    parameter = lines[0][7]
    optimizer = lines[0][2]
    learning_rate = lines[0][3]
    batch_size = lines[0][4]
    
    best_acc = 0
    best_loss = 1e8
    print(f'{arch},{num_block}({parameter}),{optimizer},{learning_rate},{batch_size},',end='')
    for i in range(10):
        if best_acc < float(lines[2*i+1][7]): best_acc = float(lines[2*i+1][7])
        if float(lines[2*i+2][7]) < best_loss: best_loss = float(lines[2*i+2][7])
        print(f'{lines[2*i+1][7]}({lines[2*i+2][7]}),', end='')
    print(f'{best_acc},{best_loss}')


fin = open('./log.txt-1680854493', 'rt')
idx = 0
tmp = []

while 1:
    line = fin.readline()
    if line == "":
        break

    line = line.strip().split(":")
    # Arch, Layer, optim, lr, batch, epoch, loss/acc, value

    if int(line[5]) % 10 == 0:
        tmp.append(line)
        
        if len(tmp) == 21:
            one_set(tmp)
            tmp = [] 

        idx += 1

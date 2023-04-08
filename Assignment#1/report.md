# AI502 Deep Learning (Spring 2023)
### Programming Assignment 1

> Hyogon Ryu, 20233477

1. **Model Architecture**

I experimented with FCNN and CNN of various architectures. As a result of the experiment, CNN using 3 blocks showed the best performance. The test accuracy at that time was 83.35%, and the train loss was 0.2842. Detailed experimental results can be found in section 4.

At this time, the block of CNN is as follows.

```python
        cfg = [3, 32, 64, 128]
        self.block = []
        for i in range(num_block):
            self.block += [nn.Conv2d(cfg[i], cfg[i+1], 3, padding=1), nn.BatchNorm2d(cfg[i+1]), nn.ReLU(True), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*self.block)
```

2. **Comparison between FCNN and CNN**

Compared to FCNN, CNN was better in test accuracy, train loss, and number of parameters. In the case of test accuracy, FCNN achieved around 60%, but CNN achieved around 80%. The train loss was 0.5293 for FCNN, but 0.2716 for CNN. Comparing the number of parameters, FCNN had the smallest number of parameters of 1,842,186, but CNN had the largest number of parameters of 357,258. It's about 6 times less than FCNN.

3. **Effects of hyperparameters**

As a hyperparameter, I tried different optimizer, learning rate, batch size, and epoch. Full experiment results can be found in section 4, table 1.

- Optimizer

In the case of the optimizer, there was a tendency to reach the best test accuracy earlier when using Adam than when using SGD.

Also, as can be seen by checking the numbers marked in blue in table 1., when Adam was used, the best test accuracy was able to achieve higher values. For 6 architectures, Adam scored 4 times higher accuracy than SGD.

- learning rate

Only when SGD was used as the optimizer, when the learning rate was set to 0.01, higher accuracy was recorded at a much earlier epoch than when the learning rate was set to 0.001. On the other hand, when the optimizer was set to Adam, there was no significant difference or it was reversed.

If you check the final best test accuracy, which is displayed in blue, the learning rate was set to 0.001 4 times out of 6 times.

* batch size

In the case of batch size, no particular tendency was found. Checking the final best test accuracy marked in blue, there were more cases with a batch size of 512, 4 out of 6.

* epoch

In the case of epoch, of course, the larger the epoch, the smaller the train loss. Therefore, the train loss was the smallest at 100 epochs except for one case.

Even in the case of test accuracy, it tended to show high epochs at high epochs. However, when checking the cases of CNN with 1 block, there were cases where the highest accuracy was achieved between epoch 20 and 40, which is presumed to be due to overfitting because of the small size of the model (small model parameters).

The epoch tended to decrease after achieving the highest value. This also appears to be due to overfitting.

* best combination

3 of the final best test accuracy, shown in blue, was the Adam, 0.001, 512 combination.

* Discussion

As for the experiments on hyperparameters, my conclusions could be wrong because I did not deal with so many cases. Much more experimentation will be needed to solidify the conclusions.

4. **Experiment Results**

<img width="1418" alt="experiment-results" src="https://user-images.githubusercontent.com/56115311/230736659-770023fa-144e-4cdf-9053-497195332fc1.png">


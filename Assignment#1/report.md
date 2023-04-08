# AI502 Deep Learning (Spring 2023)
### Programming Assignment 1

> Hyogon Ryu, 20233477

1. Model Architecture

I experimented with FCNN and CNN of various architectures. As a result of the experiment, CNN using 3 blocks showed the best performance.

The test accuracy at that time was 83.35%, and the train loss was 0.2842.

Detailed experimental results can be found in section 4.

At this time, the block of CNN is as follows.

```python
        cfg = [3, 32, 64, 128]
        self.block = []
        for i in range(num_block):
            self.block += [nn.Conv2d(cfg[i], cfg[i+1], 3, padding=1), nn.BatchNorm2d(cfg[i+1]), nn.ReLU(True), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*self.block)
```

2. Comparison between FCNN and CNN

Compared to FCNN, CNN was better in test accuracy, train loss, and number of parameters.

In the case of test accuracy, FCNN achieved around 60%, but CNN achieved around 80%.

The train loss was 0.5293 for FCNN, but 0.2716 for CNN.

Comparing the number of parameters, FCNN had the smallest number of parameters of 1,842,186, but CNN had the largest number of parameters of 357,258. It's about 6 times less than FCNN.



3. Effects of hyperparameters

hyper parameter로서 optimizer, learning rate, batch size, epoch를 달리해보았다. 전체 실험결과는 section 4, table 1. 에서 확인 가능하다.

- Optimizer

optimizer의 경우, 대체적으로 Adam을 사용했을 때가 SGD를 사용했을 때보다 best test accuracy에 먼저 도달하는 경향이 존재했다. 

또한, table 1.에서 파란색으로 표시된 숫자들을 확인해보면 알 수 있듯이, Adam을 사용했을 때, best test accuracy가 더 높은 수치를 달성할 수 있었다.  6개의 architecture에 대해서 4번 Adam이 SGD보다 높은 accuracy를 기록했다.

- learning rate

optimizer를 SGD로 했을 때만 보면, learning rate를 0.01로 했을 때가 0.001로 했을 때에 비해 훨씬 초반 epoch에 더 높은 accuracy를 기록했다. 반면에 optimizer를 Adam으로 했을 때는 큰 차이가 없거나 역전되었다. 

파란색으로 표시된 최종 best test accuracy를 확인해보면, learning rate를 0.001로 한 경우가 6번 중 4번으로 더 많았다. 

* batch size

batch size의 경우에는 큰 경향성을 찾는 것이 어려웠다.

파란색으로 표시된 최종 best test accuracy를 확인해보면, batch size를 512로 한 경우가 6번 중 4번으로 더 많았다. 

* epoch

epoch의 경우, 당연하게도 크면 클수록 train loss가 줄어들었다. 그렇기에 한번의 경우를 제외하면 모두 100 epoch에서 train loss가 가장 작았다.

test accuracy의 경우에도 높은 epoch에서 높은 epoch를 나타내는 경향이 있었다. 그러나 CNN, 1block의 케이스를 확인해보면 20~40에서 가장 높은 accuracy를 달성하는 경우들이 존재하였는데 이는 model의 사이즈가 작아서(model parameter가 작아서) overfitting이 발생한 것으로 추정된다.

epoch은 가장 높은 수치를 달성한 이후에는 낮아지는 경향성을 보였다. 이역시 overfitting으로 인한 것으로 보인다.

* best combination

파란색으로 표시된 최종 best test accuracy 중 3번이 Adam, 0.001, 512 조합이었다.

* Discussion

Hyperparameter에 대한 실험은 내가 진행한 것이 그렇게 많은 케이스를 다루고 있는 것이 아니기에, 나의 결론이 틀릴 수도 있다. 결론을 더 탄탄하게 하려면 훨씬 더 많은 실험을 진행해야할 것이다. 

4. Experiment Results




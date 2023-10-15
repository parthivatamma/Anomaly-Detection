# Anomaly-Detection

Anomaly Detection using the isolation forest method
- Create a training dataset and a test dataset using a file creator with random values
- trained the model using the training dataset
- tested using the test dataset

How to run
- Run AnomalyDetection.py
Sample output:
    Overs  Scores
0       1      15
1       2      10
2       3      17
3       4      10
4       5      12
5       6      20
6       7      12
7       8       7
8       9       8
9      10      11
10     11      15
11     12      14
12     13       3
13     14      19
14     15      11
15     16      13
16     17      40
17     18      16
18     19      26
19     20      30
20     21       0
    Overs  Scores    scores  anomaly
0       1      29 -0.033180       -1
1       2      17  0.162398        1
2       3      13  0.135136        1
3       4      32 -0.066483       -1
4       5       7  0.091210        1
5       6       6  0.027284        1
6       7      30 -0.041341       -1
7       8       4 -0.010490       -1
8       9      26  0.000000        1
9      10       3 -0.041341       -1
10     11       9  0.144382        1
11     12      20  0.080467        1
12     13      25  0.007618        1
13     14       4 -0.010490       -1
14     15       0 -0.092642       -1
15     16      26  0.000000        1
16     17      21  0.080467        1
17     18      11  0.181823        1
18     19      12  0.171154        1
19     20      15  0.185300        1
7

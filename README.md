# 2016 CV final project
team : 19

Introduction:
As we know that the fine-tune problem is quite a
challenge because the detailed pattern and feature
recognition is complex and difficult. And there are many
dogs in our campus that we cannot recognize them
immediately. We decided to combine these two issues and
proposed two methods to distinguish dogs in our campus.
The first method we extract the features by using HoG and
then classify by SVM. The second method we construct a
CNN classification model which directly extracts the
features without extra techniques such as SIFT and HoG.
With this network, we can easily tell the dogs with only a
single picture of the dog and get the information about
these dogs in detail.

step1:
data_augmetation.m
To increase the training data.

step2:
select the method:
(1) HoG + SVM
run the main.m

(2) CNN
a.use data_preprocessing.m
b.run the python file

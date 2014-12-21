pypy tradeshift.py
vw -d ../data/train_y33.vw -f model_y33.vw --loss_function logistic \
  -b 28 -l 0.7 -q hh -q hb -q hp -q hr -c -k --passes 5 \
  --hash all --random_seed 42
vw -d ../data/test.vw -t -i model_y33.vw -p preds_y33.p.txt
pypy vw_to_submission.py

--rank 100
--nn 10
--l1 1.0
--l2 1.0
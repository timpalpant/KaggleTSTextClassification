./one_hot_all.py ../../data/train.train.npz ../../data/train.train.one_hot_all.vw
./one_hot_all.py ../../data/train.validate.npz ../../data/train.validate.one_hot_all.vw
for i in {0..32}; do 
    ./encode_for_vw.py ../../data/train.train.one_hot_all.vw ../../data/trainLabels.npz  ../../data/train.train.one_hot_all.y${i}.vw --label ${i}
    vw -d ../../data/train.train.one_hot_all.y${i}.vw -f model_y${i}.vw --loss_function logistic -b 28 -l 0.7 -q ss -q sb -q sf -q si -c --passes 3 --hash all --random_seed 42 --compressed;
    vw -d ../../data/train.validate.one_hot_all.vw -i model_y${i}.vw -p preds_y${i}.p.txt --compressed -c; 
    ./score_vw.py preds_y${i}.p.txt ../../data/trainLabels.validate.npz --label ${i}
done

do_vw() {
  echo Fitting label $1;
  ./encode_for_vw.py ../../data/train.one_hot_all_grouped.vw ../../data/trainLabels.npz  ../../data/train.one_hot_all_grouped.y${1}.vw --label ${1}
  # train model with L1 regularization (LASSO)
  vw -d ../../data/train.one_hot_all_grouped.y${1}.vw -f model_y${1}.grouped.l1.vw --loss_function logistic -b 28 -l 0.3 -q bb -q jj -q bj -q bc -q ab -q dj -q db -q cc -q dd --passes 3 --hash all --random_seed 42 --compressed -c --l1 1e-8;
  # re-train model with LASSO-selected features
  vw -d ../../data/train.one_hot_all_grouped.y${1}.vw -f model_y${1}.grouped.vw --loss_function logistic -b 28 -l 0.3 -q bb -q jj -q bj -q bc -q ab -q dj -q db -q cc -q dd --passes 3 --hash all --random_seed 42 --compressed -c --feature_mask model_y${1}.grouped.l1.vw;
  rm ../../data/train.one_hot_all_grouped.y${1}.vw
  rm ../../data/train.one_hot_all_grouped.y${1}.vw.cache
  rm model_y${1}.grouped.l1.vw
  echo Predicting label $1;
  vw -d ../../data/test.one_hot_all_grouped.vw -i model_y${1}.grouped.vw -p preds_y${1}.grouped.p.txt --compressed -c; 
  rm -f model_y${1}.grouped.vw
}

do_vw() {
  echo Fitting label $1;
  ./encode_for_vw.py ../../data/train.one_hot_all.vw ../../data/trainLabels.npz  ../../data/train.one_hot_all.y${1}.vw --label ${1}
  vw -d ../../data/train.one_hot_all.y${1}.vw -f model_y${1}.l1.vw --loss_function logistic -b 28 -l 0.3 -q ss -q sb -q sf -q si -q bb -q bf -q bi -q ff -q fi -q ii --cubic sss --passes 3 --hash all --random_seed 42 --compressed -c --l1 1e-8;
  vw -d ../../data/train.one_hot_all.y${1}.vw -f model_y${1}.vw --loss_function logistic -b 28 -l 0.3 -q ss -q sb -q sf -q si -q bb -q bf -q bi -q ff -q fi -q ii --cubic sss --passes 3 --hash all --random_seed 42 --compressed -c --feature_mask model_y${i}.l1.vw;
  rm ../../data/train.one_hot_all.y${1}.vw
  rm ../../data/train.one_hot_all.y${1}.vw.cache
  echo Predicting label $1;
  vw -d ../../data/test.one_hot_all.vw -i model_y${1}.vw -p preds_y${1}.p.txt --compressed -c; 
  rm -f model_y${1}.vw
}

export -f do_vw
seq 0 32 | parallel -j 2 do_vw {}

--rank 100
--nn 10
--l1 1e-8
--l2 1e-8

a: best floats
b: best hashes
c: best bools
d: last (best) group of ints
e-h: other ints
i: other floats
j: other strings
k: other bools

Score: 0.0548616895752
vw -d ../../data/train.train.one_hot_all_grouped.y32.vw -f model_y32.grouped.vw --loss_function logistic -b 28 -l 0.7 -q bb -q jj -q bj -q bc -q ab -q dj -q db -q cc -q dd -c --passes 3 --hash all --random_seed 42 --compressed

average    since         example     example  current  current  current
loss       last          counter      weight    label  predict features
0.693147   0.693147            1         1.0   1.0000   0.0000      500
0.953516   1.213886            2         2.0  -1.0000   0.8614      500
0.813444   0.673372            4         4.0   1.0000  -0.0555      500
0.650873   0.488301            8         8.0   1.0000   0.4329      500
0.686365   0.721857           16        16.0   1.0000   0.2004      500
0.620876   0.555387           32        32.0   1.0000   0.6568      500
0.617622   0.614367           64        64.0   1.0000   0.6527      500
0.571368   0.525115          128       128.0   1.0000  -0.5832      500
0.528239   0.485109          256       256.0  -1.0000  -3.3378      500
0.497619   0.467000          512       512.0   1.0000   2.9141      500
0.442738   0.387857         1024      1024.0   1.0000   0.4429      500
0.405625   0.368512         2048      2048.0  -1.0000  -4.5799      500
0.365149   0.324672         4096      4096.0  -1.0000  -3.8774      500
0.321752   0.278355         8192      8192.0   1.0000  -0.1637      500
0.279518   0.237284        16384     16384.0  -1.0000  -2.4155      500
0.237200   0.194882        32768     32768.0   1.0000   2.6518      500
0.199480   0.161761        65536     65536.0   1.0000   5.0570      500
0.165010   0.130540       131072    131072.0   1.0000   6.2495      500
0.136246   0.107483       262144    262144.0  -1.0000  -5.4601      500
0.111109   0.085972       524288    524288.0   1.0000   6.8535      500
0.090683   0.070256      1048576   1048576.0   1.0000   5.0823      500
0.074348   0.074348      2097152   2097152.0   1.0000   7.7164      500 h
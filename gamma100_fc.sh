AGG=1
for LL in 2
do
for CC in 0.25 1
do
  echo "python fc_margin.py --lbda ${LL} --C ${CC} --aggregate ${AGG} --anneal_epoch 0 --epochs 20 | tee fc_log/lbda${LL}_C${CC}_agg${AGG}.log"
  python fc_margin.py --lbda ${LL} --C ${CC} --aggregate ${AGG} --anneal_epoch 0 --epochs 20 | tee fc_log/lbda${LL}_C${CC}_agg${AGG}.log
done
  echo ""
done

# Vanilla
LL=0
CC=0  
echo "python fc_margin.py --lbda ${LL} --C ${CC} --aggregate ${AGG} --anneal_epoch 0 --epochs 20 | tee fc_log/lbda${LL}_C${CC}_agg${AGG}.log"
python fc_margin.py --lbda ${LL} --C ${CC} --aggregate ${AGG} --anneal_epoch 0 --epochs 20 | tee fc_log/lbda${LL}_C${CC}_agg${AGG}.log


echo "START 1"
python finetune.py --device cuda:0 --Exp_N 1 &
echo "START 2"
python finetune.py --device cuda:1 --Exp_N 2 &
echo "START 3"
python finetune.py --device cuda:2 --Exp_N 3 &
echo "START 4"
python finetune.py --device cuda:3 --Exp_N 4 &
echo "START 5"
python finetune.py --device cuda:4 --Exp_N 5 &
echo "START 6"
python finetune.py --device cuda:5 --Exp_N 6 &
echo "START 7"
python finetune.py --device cuda:6 --Exp_N 7 &
echo "START 8"
python finetune.py --device cuda:7 --Exp_N 8 &

# Python Environment Setup #
We use [nlg-eval](https://github.com/Maluuba/nlg-eval) evaluation toolkit to measure the performance of our models on different tasks. The following scripts are only tested with Python 3.6 & 3.7 on Ubuntu 18.04 and MacOS.
```bash
# Assuming you have Java 1.8.0 (or higher) installed already
pip install numpy
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```


# Evaluate Results #
Each line in *ref.txt* is a ground truth answer.  
Each line in *pred.txt* is a predicted answer.  
The number of lines in *ref.txt* is twice as many as the number in *pred.txt*, because each predicted answer corresponds to two references.

```bash
python evaluate_test_results.py     \
    --target_file=ref.txt           \
    --result_file=pred.txt
```

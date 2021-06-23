# Data Format #

Data examples for training, validation and testing can be found in *data* and *data_for_fid* folders. The *.source* and *.target* files are paired with line-to-line matching. Each line is one data sample. The *.source* files are the inputs to the ; the *.target* files are the outputs.

Each data sample for **FiD** contains multiple segments. They are separated by "\t" in each line.


# Notes #
- When applying **FiD** to the data, make sure the number of segments across different data samples are the same if one wants to train with `batch_size > 1`; otherwise, one can only train a model with `batch_size = 1`. It relates to some issues with a "segment"-padded tensor.  Future update will provide better flexibility.
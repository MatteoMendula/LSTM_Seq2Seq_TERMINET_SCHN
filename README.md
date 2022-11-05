# LSTM_Seq2Seq_TERMINET_SCHN

web server location: consegna/docker deployment

POST APIs:
1. predict_on_sequence
2. add_new_data
3. retrain

predict_on_sequence expects a 35x3 matrix and returns a 22x3 matrix
the input parameter (JSON) is called "sequence" in output always returns a JSON with two parameters "input_seq" (the received sequence) and "output_seq" (the predicted sequence)

add_new_data expects a "new_data" parameter which consists of a list of strings, these are added to the bottom of the file on which the training takes place

retrain does not want any parameters and simply forces a new training with the same architecture tested by me on the available data
# Just testing some methods of indexing the merged one hot encoded data
import numpy as np

if __name__ == "__main__":
    # manualy creating a dummy one hot encoded data
    # ususally we would use the one hot encoder from sklearn
    word = "BAC"
    ohe = [[0,1,0],[1,0,0],[0,0,1]]
    word_to_index = {"A":0, "B":1, "C":2}
    index_to_word = {0:"A", 1:"B", 2:"C"}
    ohe = np.array(ohe)
    merged_ohe = np.concatenate((ohe)) # now we are dealing with ([0,1,0, 1,0,0, 0,0,1])

    # we cound just use a for loop:
    encoded_word = []
    for i in range(0, len(merged_ohe), 3):
        index_key = np.argmax(merged_ohe[i:i+3])
        encoded_word.append(index_to_word[index_key])
    
    print("".join(encoded_word)) # gives BAC, we could also create the string in the for loop, 
                                 # but our ohe sequences are also represented as list


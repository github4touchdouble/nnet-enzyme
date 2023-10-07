from reading_util import load_ml_data_emb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

def correlate_embedding(prott5, esm2):

    # Calculate correlation coefficient
    corr = np.corrcoef(prott5, esm2)
    
    # Plot
    # plt.scatter(prott5, esm2)
    # plt.show()
    
    return corr[0][1]

def correlate_embedding_2(prott5, esm2):

    df = pd.DataFrame({'prott5': prott5, 'esm2': esm2})

    # Plot
    sns.jointplot(x='prott5', y='esm2', data=df, kind='reg', color='b')
    plt.show()


def read_embedings(path_to_prott5, path_to_esm2, path_to_csv):

    # Read ESM2
    X_esm2, _ = load_ml_data_emb(path_to_esm2, path_to_csv)
    del _


    # Read ProtT5
    X_prott5, _ = load_ml_data_emb(path_to_prott5, path_to_csv)
    del _


    return X_esm2, X_prott5


if __name__ == "__main__":

    # Load environment variables
    load_dotenv()
    path_to_prott5 = os.getenv("PROTT5_ENZYMES_SPLIT_10")
    path_to_esm2 = os.getenv("ESM2_ENZYMES_SPLIT_10")
    path_to_csv = os.getenv("CSV10_ENZYMES")

    # Read embeddings
    esm2, prott5 = read_embedings(path_to_prott5, path_to_esm2, path_to_csv)

    
    # Correlate embeddings
    pearsons_corr = []
    for i in range(len(esm2)):
        pearsons_corr.append(correlate_embedding(esm2[i][:1024], prott5[i]))

    # Plot
    sns.scatterplot(pearsons_corr)
    plt.title("Pearson's Correlation for Subsets of esm2 and prott5")
    plt.xlabel("Data Points")
    plt.ylabel("Pearson's Correlation Coefficient")
    plt.show()

    print(f"Highest pearson correlation: {max(pearsons_corr)}")
    print(f"Lowes pearson correlation: {min(pearsons_corr)}")






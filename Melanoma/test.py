import numpy as np
import pandas as pd
if __name__ == "__main__":
    df_rand=pd.DataFrame(np.random.randint(0,3,size=(100,4)),columns=list('ABCD'))
    print(df_rand.head())
    print(df_rand.A.mode()[0])
    print(np.random.randint(0,10,10))
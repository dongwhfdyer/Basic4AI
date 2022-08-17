import numpy as np
import pandas as pd

# init one dataframe with random numbers
df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
df['rubb'] = 0

print("hello")

# Full feature set except Whois + Blacklist
import numpy as np

Hostbased_Feature_path = 'Host_based_FeatureSet.npy'
Full_ex_wb_path = 'Full_except_WB.npy'
Lexical_Feature_path = 'Lexical_FeatureSet.npy'

hostbased = np.load(Hostbased_Feature_path)
Hostbased_ex_wb = hostbased[:,6:]
lexical = np.load(Lexical_Feature_path)

Full_ex_wb = np.hstack((lexical,Hostbased_ex_wb))
print(Full_ex_wb.shape)
# (7000,13)

np.save(Full_ex_wb_path,Full_ex_wb)
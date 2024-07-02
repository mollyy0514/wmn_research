from myutils import *
from ho_profile import *

dirc_mets_list = ['dl_lost', 'ul_lost', 'dl_excl', 'ul_excl']
epochs = 2
test_mode = False

## ---------- BROWN LINE ---------- ##
## create a new model name / load a model name
# model_id = model_identity()
model_id = "20240620_1537000709"
model_dscp = 'new_data_sync_v2'
print('Model ID:', model_id, model_dscp)

## load data
# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = ['2024-05-23']
selected_exps = []
selected_routes = ['BR']
exluded_dates = []
excluded_exps =[]
excluded_routes = []
filepaths = data_loader(mode='sr', show_info= True, selected_dates=selected_dates, selected_routes=selected_routes)
print(len(filepaths))

## Single radio profle
# for dirc_mets in dirc_mets_list:
#     model = Profile(filepaths, 'BR', model_id, model_dscp, dirc_mets=dirc_mets, epochs=epochs, test_mode=test_mode)

## Single radio evaluate
eval = Eval(filepaths, 'BR', model_id, model_dscp, dirc_mets='dl_lost', save_answer=True)
eval.run_hist_method(N=1)
eval.plot()
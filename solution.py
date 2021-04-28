import pandas as pd
import numpy as np
from os.path import join
import py_entitymatching as em

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)

    return LR


def block_by_brand(ltable, rtable):
    em.set_key(ltable, 'id')
    em.set_key(rtable, 'id')
    ob = em.OverlapBlocker()
    cand_set = ob.block_tables(ltable, rtable, 'title', 'title', word_level=True, overlap_size=4,
                               l_output_attrs=['id', 'title', 'category', "brand", 'modelno', 'price'],
                               r_output_attrs=['id', 'title', 'category', "brand", 'modelno', 'price'],
                               show_progress=False, rem_stop_words=True)
    ab = em.AttrEquivalenceBlocker()
    double_cs = ab.block_candset(cand_set, 'modelno', 'modelno', show_progress=False, allow_missing=False)
    # double_cs = ab.block_candset(double_cs, 'brand', 'brand', show_progress=False, allow_missing=True)
    '''cand_set = ab.block_tables(ltable, rtable, 'brand', 'brand',
                               l_output_attrs=['id', 'title', 'category', "brand", 'modelno', 'price'],
                               r_output_attrs=['id', 'title', 'category', "brand", 'modelno', 'price'],
                               )
    double_cs = ob.block_candset(cand_set, 'modelno', 'modelno', show_progress=False)'''
    # double_cs = em.combine_blocker_outputs_via_union([cand_set, double_cs])
    cs_id_df = double_cs.loc[:, ['ltable_id', 'rtable_id']]
    candset = cs_id_df.values.tolist()

    return candset


# blocking to reduce the number of pairs to be compared
candset = block_by_brand(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking", len(candset))
candset_df = pairs2LR(ltable, rtable, candset)

# 3. Feature engineering
import Levenshtein as lev


def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)


def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    #print(len(features[0]))
    return features


candset_features = feature_engineering(candset_df)
# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

gb = GradientBoostingClassifier(max_features=10, n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)
gb.fit(training_features, training_label)
y_pred = gb.predict(candset_features)
rf = RandomForestClassifier(random_state=0)
rf.fit(training_features, training_label)

#y_pred = rf.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])

#print(pred_df.to_string)
pred_df.to_csv("output.csv", index=False)


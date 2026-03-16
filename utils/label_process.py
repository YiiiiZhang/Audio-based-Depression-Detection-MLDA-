import pandas as pd
from tqdm import tqdm
# ==========================================================
#                      Utility functions
# ==========================================================

def format_id(i):
    """Convert id to a 3-digit format like '007'"""
    return f"{int(i):03d}"


def tri_label(cid, healthy_set, depressed_set):
    """
    return -1 / 0 / 1:
    1  -> cid in depressed_set
    0  -> cid in healthy_set
    -1 -> other
    """
    if cid in depressed_set:
        return 1
    elif cid in healthy_set:
        return 0
    else:
        return -1
def tri_type(cid, cr_list, cradk_list):
    """
    return -1 / 0 / 1:
    1  -> cid in depressed_set
    0  -> cid in healthy_set
    -1 -> other
    """
    if cid in cr_list:
        return "CR"
    elif cid in cradk_list:
        return "CR_ADK"
    else:
        return "None"


# ==========================================================
#                    Scale Label Function
# ==========================================================

def high_HRSD(threshold=30, gender='both',
              labels_path='labels/20251105_d02_questionnaires_app.xlsx'):
    df = pd.read_excel(labels_path)

    Healthy = []
    Depressed = []

    for _, row in df.iterrows():
        score = row['HRSD_24.1']
        if score <= 8:
            Healthy.append(row['id']) #Healthy set
        elif score > threshold:
            Depressed.append(row['id']) #Depressed set
        else:
            pass  

    # Gender Filter
    if gender == 'm':
        gender_num = 1
    elif gender == 'w':
        gender_num = 2
    else:
        gender_num = None

    if gender_num is not None:
        valid_ids = df[df['gender'] == gender_num]['id'].values
        Healthy = [i for i in Healthy if i in valid_ids]
        Depressed = [i for i in Depressed if i in valid_ids]

    Healthy = set(format_id(i) for i in Healthy)
    Depressed = set(format_id(i) for i in Depressed)

    return Healthy, Depressed


def detect_Depression(
    gender='both',
    labels_path='labels/20251105_d02_questionnaires_app.xlsx'
):
    df = pd.read_excel(labels_path)

    xls_d = df[df['diag'] == 'd']['id'].values
    xls_h = df[df['diag'] == 'nd']['id'].values

    if gender == 'm':
        gender_num = 1
    elif gender == 'w':
        gender_num = 2
    else:
        gender_num = None

    if gender_num is not None:
        xls_d = [i for i in xls_d if df.loc[df['id'] == i, 'gender'].values[0] == gender_num]
        xls_h = [i for i in xls_h if df.loc[df['id'] == i, 'gender'].values[0] == gender_num]

    Healthy = set(format_id(i) for i in xls_h)
    Depressed = set(format_id(i) for i in xls_d)

    return Healthy, Depressed


def detect_symptoms(
    symptom_name='retardation',
    gender='both',
    labels_path='labels/20251105_d02_questionnaires_app.xlsx'
):
    df = pd.read_excel(labels_path)

    if symptom_name == 'retardation':
        col = 'D_HRSD_08'
    elif symptom_name == 'insomnia':
        col = 'D_HRSD_05'
    elif symptom_name == 'agitation':
        col = 'D_HRSD_09'
    elif symptom_name == 'weight_loss':
        col = 'D_HRSD_10'
    else:
        raise ValueError("invalid symptom name")

    Healthy = []
    Depressed = []

    for _, row in df.iterrows():
        v = row[col]
        if v == 0:
            Healthy.append(row['id'])
        elif v in [1, 2, 3, 4, 5]:
            Depressed.append(row['id'])
        else:
            pass  

    if gender == 'm':
        gender_num = 1
    elif gender == 'w':
        gender_num = 2
    else:
        gender_num = None

    if gender_num is not None:
        valid_ids = df[df['gender'] == gender_num]['id'].values
        Healthy = [i for i in Healthy if i in valid_ids]
        Depressed = [i for i in Depressed if i in valid_ids]

    Healthy = set(format_id(i) for i in Healthy)
    Depressed = set(format_id(i) for i in Depressed)

    return Healthy, Depressed

def detect_cradk(labels_path):
    df = pd.read_excel(labels_path)
    cr_ids = df[df['condition']=='cr']['id'].to_list()
    cradk_ids = df[df['condition']=='cradk']['id'].to_list()
    return cr_ids,cradk_ids

def build_full_dataset(
    data,
    labels_path='labels/20251105_d02_questionnaires_app.xlsx',
    hrsd_threshold=30,
    gender="both"
):

    # ======== First, get the Healthy / Depressed sets for all labels ========
    h_dep, d_dep = detect_Depression(gender, labels_path)
    h_hrsd, d_hrsd = high_HRSD(hrsd_threshold, gender, labels_path)
    h_ret, d_ret = detect_symptoms("retardation", gender, labels_path)
    h_ins, d_ins = detect_symptoms("insomnia", gender, labels_path)
    h_agi, d_agi = detect_symptoms("agitation", gender, labels_path)
    h_wl, d_wl = detect_symptoms("weight_loss", gender, labels_path)
    crlist,cradk_list = detect_cradk(labels_path)

    result = data.copy()

    for raw_id in tqdm(data.keys(),desc="Processing cases"):
        cid = format_id(raw_id)

        label_info = {
            "is_depression":   tri_label(cid, h_dep, d_dep),
            "is_HRSD":         tri_label(cid, h_hrsd, d_hrsd),
            "is_retardation":  tri_label(cid, h_ret, d_ret),
            "is_insomnia":     tri_label(cid, h_ins, d_ins),
            "is_agitation":    tri_label(cid, h_agi, d_agi),
            "is_weight_loss":  tri_label(cid, h_wl, d_wl),
            "type":            tri_type(int(cid), crlist, cradk_list)
        }
        result[cid]["label"] = label_info
    return result
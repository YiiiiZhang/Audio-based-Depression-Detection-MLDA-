def detect_symptoms(Dep_path = 'labels/labels_RCT_depressed.csv', H_path = 'labels/labels_RCT_healthy.csv', file_path = 'labels/HRSD.xlsx', symptom_name = 'retardation', gender = 'both'):

    # Read the Excel file

    df = pd.read_excel(file_path)

 

    # Initialize lists

    Healthy = []

    Depressed = []

 

    if symptom_name == 'retardation':

        symptom_column = 'D_HRSD_08'

    elif symptom_name == 'insomnia':

        symptom_column = 'D_HRSD_05'

    elif symptom_name == 'agitation':

        symptom_column = 'D_HRSD_09'   

    elif symptom_name == 'weight_loss':

        symptom_column = 'D_HRSD_10'

    else:

        raise ValueError("Invalid symptom name. Choose from 'retardation', 'insomnia', 'agitation', or 'weight_loss'.")

   

    # Loop through the DataFrame and categorize IDs

    for _, row in df.iterrows():

        if row[symptom_column] == 1:

            Healthy.append(row['id'])

        elif row[symptom_column] in [2, 3, 4, 5]:

            Depressed.append(row['id'])

        else:

            pass

 

    Healthy = [int(x) for x in Healthy]

    nHealthy = []

    Depressed  = [int(x) for x in Depressed]

    nDepressed = []

 

    # selecting healthy and depressed patients that are male of female based on Depressed_gender and Healthy_gender

    xls_d = pd.read_csv(Dep_path)

    xls_h = pd.read_csv(H_path)

 

    if gender == 'both':

        pass

    else:

        Depressed_gender = xls_d[xls_d['gender'] == gender]['ID'].values

        Healthy_gender = xls_h[xls_h['gender'] == gender]['ID'].values

        Full_gender = list(set(Depressed_gender) | set(Healthy_gender))

        print(len(Full_gender))

        print(len(Depressed))

        print(len(Healthy))

        Healthy_new = [x for x in Healthy if x in Full_gender]

        Depressed_new = [x for x in Depressed if x in Full_gender]

        Healthy = Healthy_new

        Depressed = Depressed_new

 

    for i in Healthy:

        j = "00" + str(i)

        if len(j) == 6:

            k = j[-4:]

        else:

            k = j[-3:]

 

        nHealthy.append(k)

 

    for i in Depressed:

        j = "00" + str(i)

        if len(j) == 6:

            k = j[-4:]

        else:

            k = j[-3:]

        nDepressed.append(k)

 

    Healthy = nHealthy

    Depressed = nDepressed

 

    return Healthy, Depressed  

 

def high_HRSD(threshold = 30, Dep_path = 'labels/labels_RCT_depressed.csv', H_path = 'labels/labels_RCT_healthy.csv', gender = 'both'):

    # Read the Excel file

    df = pd.read_excel('labels/d02_questionnaires_app.xlsx')

 

    # Initialize lists

    Healthy = []

    Depressed = []

 

    # Loop through the DataFrame and categorize IDs

    for _, row in df.iterrows():

        if row['HRSD_24.1'] <= threshold:

            Healthy.append(row['id'])

        elif row['HRSD_24.1'] > threshold:

            Depressed.append(row['id'])

        else:

            pass

 

    Healthy = [int(x) for x in Healthy]

    nHealthy = []

    Depressed  = [int(x) for x in Depressed]

    nDepressed = []

 

    # selecting healthy and depressed patients that are male of female based on Depressed_gender and Healthy_gender

    xls_d = pd.read_csv(Dep_path)

    xls_h = pd.read_csv(H_path)

 

    if gender == 'both':

        pass

    else:

        Depressed_gender = xls_d[xls_d['gender'] == gender]['ID'].values

        Healthy_gender = xls_h[xls_h['gender'] == gender]['ID'].values

        Full_gender = list(set(Depressed_gender) | set(Healthy_gender))

        Healthy_new = [x for x in Healthy if x in Full_gender]

        Depressed_new = [x for x in Depressed if x in Full_gender]

        Healthy = Healthy_new

        Depressed = Depressed_new

 

    for i in Healthy:

        j = "00" + str(i)

        if len(j) == 6:

            k = j[-4:]

        else:

            k = j[-3:]

 

        nHealthy.append(k)

 

    for i in Depressed:

        j = "00" + str(i)

        if len(j) == 6:

            k = j[-4:]

        else:

            k = j[-3:]

        nDepressed.append(k)

 

    Healthy = nHealthy

    Depressed = nDepressed

 

    return Healthy, Depressed
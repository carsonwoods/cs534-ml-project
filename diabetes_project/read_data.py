"""
Handles reading the data using a consistent interface.
"""
# 1st party imports
from pathlib import Path

# 3rd party imports
import pandas as pd

def get_data_df():
    """
    Reads CSV into pandas dataframe from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path, na_values=["?"], low_memory=False)


def get_data_numpy():
    """
    Reads CSV into numpy array from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path, na_values=["?"], low_memory=False).to_numpy()

def get_data_quality_cont():
    """
    Returns data quality report for continous variables 
    """
    df = get_data_df()
    columns = ['Feature', 'Desc.','Count','% of Missing','Card.', 'Mode','Mode Freq.','Mode %','2nd Mode','2nd Mode Freq.',	'2nd Mode %', 'Notes']
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = [], [], [], [], [], [],[],[],[],[],[], []
    df_con = df[['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses' ]]
    descs = ['Unique identifier of an encounter',
            'Unique identifier of a patient',
            'Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available',
            'Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available',
            'Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital',
            'Integer number of days between admission and discharge',
            'Number of lab tests performed during the encounter',
            'Number of procedures (other than lab tests) performed during the encounter',
            'Number of distinct generic names administered during the encounter',
            'Number of outpatient visits of the patient in the year preceding the encounter',
            'Number of emergency visits of the patient in the year preceding the encounter',
            'Number of inpatient visits of the patient in the year preceding the encounter',
            'Number of diagnoses entered to the system']
    i=0
    for (name, series) in df_con.items():
        c1.append(name)
        c2.append(descs[i])
        c3.append(series.size)
        c4.append(round(((series.isin(['?']).sum() + series.isnull().sum()) / series.size) * 100, 2))
        c5.append(series.unique().size)
        c6.append(series.mode()[0])
        c7.append(series.value_counts().iloc[0])
        c8.append(round((series.value_counts().iloc[0]/series.size)*100, 2))
        c9.append(series.value_counts().index.tolist()[1])
        c10.append(series.value_counts().iloc[1])
        c11.append(round((series.value_counts().nlargest(2).min()/series.size)*100,2))
        i+=1


    data = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    df_new2 = pd.DataFrame(data).transpose()
    df_new2.columns = columns
    df_new2

    #Rounding to 2 decimal places
    df_new2['Mode'] = df_new2['Mode'].apply(lambda x:round(x,2))
    df_new2['2nd Mode'] = df_new2['2nd Mode'].apply(lambda x:round(x,2))

    #null values for Notes
    df_new2['Notes'] = ""
    return df_new2


def get_data_quality_cat():
    """
    Returns data quality report for categorical variables 
    """
    df = get_data_df()
    cols = ['Feature', 'Desc.','Count','% of Missing','Card.', 'Mode','Mode Freq.','Mode %','2nd Mode','2nd Mode Freq.',	'2nd Mode %', 'Notes']
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12 = [], [], [], [], [], [],[],[],[],[],[], []
    df_cat = df[['race', 'gender', 'age', 'weight' , 'payer_code' , 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']]
    description = [
            'Values: Caucasian, Asian, African American, Hispanic, and other',
            'Values: male, female, and unknown/invalid	',
            'Grouped in 10-year intervals: [0, 10), [10, 20),..., [90, 100)	',
            'Weight ranges in pounds (ex: [75-100), [100-125)...)',
            'Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay',
            'Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon',
            'The primary diagnosis (coded as first three digits of ICD9); 848 distinct values',
            'Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values',
            'Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values',
            'Indicates the range of the result or if the test was not taken. Values: >200, >300, normal, and none if not measured',
            'Indicates the range of the result or if the test was not taken. Values: >8 if the result was greater than 8%, >7 if the result was greater than 7% but less than 8%, normal if the result was less than 7%, and none if not measured.',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed',
            'Indicates if there was a change in diabetic medications (either dosage or generic name). Values: change and no change',
            'Indicates if there was any diabetic medication prescribed. Values: yes and no',
            'Days to inpatient readmission. Values: <30 if the patient was readmitted in less than 30 days, >30 if the patient was readmitted in more than 30 days, and No for no record of readmission.'
                ]
    j = 0
    for (name, series) in df_cat.items():
        d1.append(name)
        d2.append(description[j])
        d3.append(series.size)
        d4.append(round(((series.isin(['?']).sum() + series.isnull().sum()) / series.size) * 100, 2))
        d5.append(series.unique().size)
        d6.append(series.mode()[0])
        d7.append(series.value_counts().iloc[0])
        d8.append(round((series.value_counts().iloc[0]/series.size)*100, 2))
        index_list = series.value_counts().index.tolist()
        d9.append(index_list[1] if len(index_list) > 1 else None)

        #   d9.append(series.value_counts().index.tolist()[1])
        value_counts = series.value_counts()
        d10.append(value_counts.iloc[1] if len(value_counts) > 1 else None)

        #   d10.append(series.value_counts().iloc[1])
        d11.append(round((series.value_counts().nlargest(2).min()/series.size)*100,2))
        j+=1

    data_cat = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]
    df_cat2 = pd.DataFrame(data_cat).transpose()
    df_cat2.columns = cols
    df_cat2['Notes'] = ""
    return df_cat2


def save_data_quality_reports():
    """
    Saves data quality reports to CSV files
    """
    df_cat = get_data_quality_cat()
    df_cont = get_data_quality_cont()

    # Save categorical data quality report to a CSV file
    cat_report_path = Path("./data/data_quality_cat_report.csv")
    df_cat.to_csv(cat_report_path, index=False)
    print(f"Categorical Data Quality Report saved to: {cat_report_path}")

    # Save continuous data quality report to a CSV file
    cont_report_path = Path("./data/data_quality_cont_report.csv")
    df_cont.to_csv(cont_report_path, index=False)
    print(f"Continuous Data Quality Report saved to: {cont_report_path}")




if __name__ == "__main__":
    print(get_data_numpy())
    save_data_quality_reports()
    print(get_data_quality_cat())
    print(get_data_quality_cont())


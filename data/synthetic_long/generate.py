import numpy as np
import pandas as pd

from tqdm import tqdm

possible_values = {
        "education": ["nessuno", "diploma", "triennale", "magistrale", "phd"],
        "job": ["disoccupato", "operaio", "privato", "impiegato", "manager", "ceo"],
        "relationship": ["single", "sposato/a", "divorziato/a", "vedovo/a"],
        "sex": ["male", "female"],
        "income": [10000,200000],
        "credit": range(0,50000),
        "savings": [5000, 200000],
        "housing": ["none", "own", "rent"],
        "loan": ["good", "bad"],
        "native_country": ["italia", "europa", "extra-europa"],
        "purpose": ["education", "furniture", "car", "house"],
        "credit_score": [0,1,2,3,4,5,6],
        "monthly_expenses": [4000, 10000]
}

mapping = {
        "education": {
            "nessuno": [0.59, 0.1, 0.1, 0.1, 0.1, 0.01],
            "diploma": [0.05, 0.3, 0.3, 0.2, 0.1, 0.05],
            "triennale": [0.01, 0.2, 0.2, 0.2, 0.29, 0.1],
            "magistrale": [0.01, 0.09, 0.3, 0.3, 0.2, 0.1],
            "phd": [0.01, 0.01, 0.19, 0.19, 0.3, 0.3]
        },
        "job": {
            "disoccupato": 9000,
            "operaio": 40000,
            "impiegato": 60000,
            "privato": 50000,
            "manager": 80000,
            "ceo": 100000
        },
        "savings": {
            "disoccupato": 9000,
            "operaio": 20000,
            "impiegato": 30000,
            "privato": 40000,
            "manager": 90000,
            "ceo": 100000
        },
        "monthly_expenses": {
            "disoccupato": 500,
            "operaio": 900,
            "impiegato": 1250,
            "privato": 1500,
            "manager": 3000,
            "ceo": 4000
        },
        "housing": ["none", "rent", "own"],
        "loan": ["good", "bad"]
}

def get_monthly_expenses(job):
    exp = int(abs(np.random.normal(mapping.get("monthly_expenses").get(job), 100)))
    return exp

def get_credit_score():
    return np.random.choice(possible_values.get("credit_score"), 1)[0]

def get_purpose(education):

    if education != "phd":
        return np.random.choice(possible_values.get("purpose"), 1, p=[0.0, 0.33, 0.34, 0.33])[0]
    elif education != "nessuno":
        return np.random.choice(possible_values.get("purpose"), 1, p=[0.4, 0.2, 0.2, 0.2])[0]
    else:
        return np.random.choice(possible_values.get("purpose"), 1, p=[0.25, 0.25, 0.25, 0.25])[0]

def get_savings(job):

    if job == "disoccupato":
        return mapping.get("job").get(job)

    income = int(abs(np.random.normal(mapping.get("job").get(job), 10000)))

    income = max(possible_values.get("savings")[0], income)
    income = min(possible_values.get("savings")[1], income)
    return income

def get_country():
    return np.random.choice(possible_values.get("native_country"), 1)[0]

def get_sex():
    return np.random.choice(possible_values.get("sex"), 1)[0]

def get_relationship(age):
    if age < 25:
        return np.random.choice(possible_values.get("relationship"), 1, p=[0.5, 0.4, 0.1, 0])[0]
    elif 25 <= age < 45:
        return np.random.choice(possible_values.get("relationship"), 1, p=[0.2, 0.4, 0.3, 0.1])[0]
    elif 45 <= age < 65:
        return np.random.choice(possible_values.get("relationship"), 1, p=[0.1, 0.4, 0.3, 0.2])[0]
    else:
        return np.random.choice(possible_values.get("relationship"), 1, p=[0.0, 0.3, 0.3, 0.2])[0]

def get_age():
    return np.random.choice(range(30,60), 1)[0]

def get_education():
    return np.random.choice(possible_values.get("education"), 1, p=[0.20, 0.20, 0.20, 0.20, 0.20])[0]

def get_job(education, det=False):

    if det:
        if education == "phd":
            return "manager"
        elif education == "laurea":
            return "impiegato"
        else:
            return "operaio"

    return np.random.choice(possible_values.get("job"), 1, p=mapping.get("education").get(education))[0]

def get_income(job, det=False):

    # When deterministic, we add this amount
    # to what the user was getting before
    if det:
        if job == "manager":
            return 40000
        elif job == "impiegato":
            return 20000
        else:
            return 10000

    if job == "disoccupato":
        return mapping.get("job").get(job)

    income = int(abs(np.random.normal(mapping.get("job").get(job), 10000)))

    income = max(possible_values.get("income")[0], income)
    income = min(possible_values.get("income")[1], income)
    return income


def get_housing(income):
    if income < 20000:
        p = [0.4, 0.4, 0.2]
    elif income < 40000:
        p = [0.01, 0.7, 0.29]
    else:
        p = [0.001, 0.299, 0.7]
    return np.random.choice(possible_values.get("housing"), 1, p=p)[0]

def get_credit(income):
    return np.random.choice(possible_values.get("credit"),1)[0]

def get_loan(job, education, income, credit, housing, relation, savings, purpose, credit_score, exp):

    if education == "triennale" or education == "magistrale" or education == "phd":
        if job == "impiegato" or job == "private":
            if income >= 50000 and credit_score > 4:

                if exp*12/income > 0.6:
                    return "bad"

                if housing == "own":

                    if savings > 50000:
                        return "good"

                    if savings > 30000:
                        if purpose in ["education", "furniture", "car"]:
                            return "good"

                else:

                    if savings > 80000:
                        return "good"

                    if savings > 50000:
                        if purpose in ["education", "furniture", "car"]:
                            return "good"

                    if savings > 30000:
                        if purpose in ["education", "furniture"]:
                            return "good"

        elif job == "manager" or job == "ceo":
            if income >= 80000 and credit_score > 2:

                if exp*12/income > 0.8:
                    return "bad"

                if savings > 60000:
                    return "good"

                if savings > 30000:
                    if purpose in ["education", "furniture", "car"]:
                        return "good"

                if savings > 20000:
                    if purpose in ["education", "furniture"]:
                        return "good"

                return "good"

    return "bad"

def generate_example():
    sex = get_sex()
    age = get_age()
    ed = get_education()
    native_country = get_country()
    relationship = get_relationship(age)
    job = get_job(ed)
    inc = get_income(job)
    hous = get_housing(inc)
    cre = get_credit(inc)
    savings = get_savings(job)
    purpose = get_purpose(ed)
    credit_score = get_credit_score()
    exp = get_monthly_expenses(job)
    loan = get_loan(job, ed, inc, cre, hous, relationship, savings, purpose, credit_score, exp)

    return age, sex, relationship, native_country, ed, job, inc, hous, cre, savings, purpose, credit_score, exp, loan


if __name__ == "__main__":

    results = []
    results_test = []

    np.random.seed(2021)

    failed = 5000
    correct = 5000

    print("[*] Sampling failed")
    while failed >= 0 or correct >= 0:
        age, sex, relationship, native_country, ed, job, inc, hous, cre, sav, purp, credit_score, exp, loan = generate_example()

        if loan == "bad" and failed >= 0:
            results.append([
                age, ed, job, inc, hous, cre, sex, relationship, native_country, sav, purp, credit_score, exp, loan
            ])
            failed -= 1


        if loan == "good" and correct >= 0:
            results.append([
                age, ed, job, inc, hous, cre, sex, relationship, native_country, sav, purp, credit_score, exp, loan
            ])
            correct -= 1

        print(failed, correct)

    for i in tqdm(range(0, 3000)):

        age, sex, relationship, native_country, ed, job, inc, hous, cre, sav, purp, credit_score, exp, loan = generate_example()

        results_test.append([
            age, ed, job, inc, hous, cre, sex, relationship, native_country, sav, purp, credit_score, exp, loan
        ])

    df = pd.DataFrame(results, columns=["age", "education", "job", "income", "house", "credit", "sex", "relationship", "country", "savings", "purpose", "credit_score", "expenses", "loan"])
    print(len(df[df.loan == "good"]), len(df[df.loan == "bad"]))
    df.to_csv("data/synthetic_long/train.csv", index=False)

    df = pd.DataFrame(results_test, columns=["age", "education", "job", "income", "house", "credit", "sex", "relationship", "country", "savings", "purpose", "credit_score", "expenses", "loan"])
    print(len(df[df.loan == "good"]), len(df[df.loan == "bad"]))
    df.to_csv("data/synthetic_long/test.csv", index=False)
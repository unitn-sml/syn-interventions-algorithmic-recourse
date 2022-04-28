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
        "housing": ["none", "own", "rent"],
        "loan": ["good", "bad"],
        "native_country": ["italia", "europa", "extra-europa"]
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
        "housing": ["none", "rent", "own"],
        "loan": ["good", "bad"]
}

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

def get_loan(job, education, income, credit, housing, relation):

    if education == "triennale" or education == "magistrale" or education == "phd":
        if job == "impiegato" or job == "private":
            if income >= 50000:
                if housing == "own":
                    return "good"
        elif job == "manager" or job == "ceo":
            if income >= 80000:
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
    loan = get_loan(job, ed, inc, cre, hous, relationship)

    return age, sex, relationship, native_country, ed, job, inc, hous, cre, loan


if __name__ == "__main__":

    results = []
    results_test = []

    np.random.seed(2021)

    failed = 5000
    correct = 5000

    print("[*] Sampling failed")
    while failed >= 0 or correct >= 0:
        age, sex, relationship, native_country, ed, job, inc, hous, cre, loan = generate_example()

        if loan == "bad" and failed >= 0:
            results.append([
                age, ed, job, inc, hous, cre, sex, relationship, native_country, loan
            ])
            failed -= 1


        if loan == "good" and correct >= 0:
            results.append([
                age, ed, job, inc, hous, cre, sex, relationship, native_country, loan
            ])
            correct -= 1

        print(failed, correct)

    for i in tqdm(range(0, 3000)):

        age, sex, relationship, native_country, ed, job, inc, hous, cre, loan = generate_example()

        results_test.append([
            age, ed, job, inc, hous, cre, sex, relationship, native_country, loan
        ])

    df = pd.DataFrame(results, columns=["age", "education", "job", "income", "house", "credit", "sex", "relationship", "country", "loan"])
    print(len(df[df.loan == "good"]), len(df[df.loan == "bad"]))
    df.to_csv("data/synthetic/train.csv", index=False)

    df = pd.DataFrame(results_test, columns=["age", "education", "job", "income", "house", "credit", "sex", "relationship", "country", "loan"])
    print(len(df[df.loan == "good"]), len(df[df.loan == "bad"]))
    df.to_csv("data/synthetic/test.csv", index=False)
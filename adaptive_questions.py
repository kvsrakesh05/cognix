from question_bank import question_bank


def is_present(value):

    if value is None:
        return False

    if value is False:
        return False

    if isinstance(value, str):

        value = value.lower().strip()

        if value in ["", "no", "false", "none", "null"]:
            return False

    return True


def get_next_question(symptoms, domain):

    if domain not in question_bank:
        return None

    for complaint_type, questions in question_bank[domain].items():

        if is_present(symptoms.get(complaint_type)):

            for field, question in questions.items():

                if field not in symptoms or symptoms[field] is None:
                    return question

            return None

    return None
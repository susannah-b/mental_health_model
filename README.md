# An analysis of mental health data of employees working in the Tech field
## Introduction
The data originates from a survey by [Open Sourcing Mental Illness](https://osmihelp.org/) of employees in tech fields. Participants were asked questions on a number of factors surrounding their workplace and mental health, which can be seen in the data section below. The goal of this analysis is to predict based on these factors, whether an employee will choose to seek mental health care. 

In this project, we will prepare the data, perform Exploratory Data Analysis, pre-process the data (including imputation of missing values with MICE), and build a tuned/cross-validated model.

The feature we are aiming to predict is 'treatment', i.e. whether or not the participant has sought treatment for a mental health issue. It is important to note the distinction between seeking treatment and having a mental health disorder; the former being only a sub-group of the latter. In addition to the ethical concerns of predicting mental health disorders in employees, by looking at whether or not an employee seeks treatment, companies can determine the factors that are most highly associated with seeking treatment to either identify stressors or examine which factors will allow an employee to seek treatment if they need it.

The explanation of each column in the data set is given below:
## Data
- Timestamp: Time the data was taken
- Age: Age of the participant
- Gender: Gender of the participant
- Country: Country the participant lives in
- state: If you live in the United States, which state or territory do you live in?
- self_employed: Are you self-employed?
- family_history: Do you have a family history of mental illness?
- treatment: Have you sought treatment for a mental health condition?
- work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
- no_employees: How many employees does your company or organization have?
- remote_work: Do you work remotely (outside of an office) at least 50% of the time?
- tech_company: Is your employer primarily a tech company/organization?
- benefits: Does your employer provide mental health benefits?
- care_options: Do you know the options for mental health care your employer provides?
- wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
- seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
- anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
- leave: How easy is it for you to take medical leave for a mental health condition?
- mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
- phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
- coworkers: Would you be willing to discuss a mental health issue with your coworkers?
- supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
- mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
- phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
- mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
- obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
- comments: Any additional notes or comments

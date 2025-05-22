print("Welcome to the Python Quiz")
name = input("What is your name? ")
print("Hi",name,",","lets begin")

score = 0

questions = [
    {
        "question": "What is the output of 2 + 2?",
        "options": ["A. 22", "B. 4", "C. 2", "D. None"],
        "answer": "B"
    },
    {
        "question": "What is the capital of France?",
        "options": ["A. London", "B. Berlin", "C. Paris", "D. Madrid"],
        "answer": "C"
    },
        {
        "question": "In Python, what function is used to output text?",
        "options": ["A. input()", "B. print()", "C. output()", "D. display()"],
        "answer": "B"
    },
    {
        "question": "What is 5 × 7?",
        "options": ["A. 25", "B. 35", "C. 45", "D. 55"],
        "answer": "B"
    },
    {
        "question": "Which country is home to the kangaroo?",
        "options": ["A. Brazil", "B. Australia", "C. South Africa", "D. India"],
        "answer": "B"
    }
]


for q in questions:
    print(q["question"])
    for option in q["options"]:
        print(option)
    
    while True:
        user_answer = input("Your answer: ").upper()
        if user_answer in ['A', 'B', 'C', 'D']:
            break
        print("Invalid input! Please enter A, B, C, or D")
        
    if user_answer == q["answer"]:
        print("Correct!\n")
        score += 1
    else:
        print("Incorrect! The correct answer is:" , q["answer"])


print("Quiz complete! Your score:", score ,"out of","5")

if score > 3:
    print("amazing!")
elif score >=4:
    print("excellent!")
elif score < 2 or 1:
    print("try again!")



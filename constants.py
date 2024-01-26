import streamlit as st
from pathlib import Path

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Read the files from the directory using pathlib
image_and_word_descriptions = {"Exercise 1": 
                                    ["A cat and dog playing on the floor of a house.", 
                                     "An uncut pepperoni pizza.", 
                                     "A woman in a black apron and blue shirt is putting icing on a cake in a kitchen."], 
                                "Exercise 2": 
                                    ["Man    Outside   Fishing", 
                                     "Car    Fast   Moves", 
                                     "Ball    Ground   Falls"], 
                                 "Exercise 3": 
                                    ["The girl walks the dog.", 
                                     "The chef cooks a meal.", 
                                     "The driver turns the wheel."]}

filepaths = []
directory = Path("./example-images")
if directory.exists():
    for file in directory.iterdir():
        if file.suffix in [".jpg", ".png", ".jpeg"]:
            filepaths.append(str(file))

systemMessage = "System Message: The following is a friendly conversation between a patient and a speech therapist specializing in treating broca's aphasia. The therapist is supportive and follows best practices from speech language therapy. The patient may be hard to understand because their speech is transcribed to text using automated text to speech software, but the therapist tries their best and asks for clarification if the text is unclear, using their understanding of how text to speech can incorrectly transcribe certain sounds to best guess what the patient is trying to say when unclear. After the patient correctly completes the exercise, the therapist will move on to the next exercise by saying, 'Let's move on to the next problem.' Ensure you only respond as the therapist and do not respond as the patient. If the patient correctly completes the exercise in one turn of conversation, move on. If the patient has not completed the exercise after five turns, move on. Do not ask for information that has already been given."
initial_ai_message = {'Exercise 1': "I'd like you to look at an image and describe what you see. Here's a description of the image:",
                      'Exercise 2': "I'd like you to form a sentence using the words I give you. Here are the words:",
                      'Exercise 3': "I'd like you to form a passive sentence using the words I give you. Here are the words:"}

initial_therapist_message = {'Exercise 1': "Let's start with this image. Can you describe what you see here?",
                             'Exercise 2': "Let's make a sentence with the following words.",
                             'Exercise 3': "Make the following into a passive sentence."}

exercise_1_examples = f"""Exercise Description: This exercise is focused on image description. We will start with a simple image and then proceed to a more complex one.

    Example conversation 1:
    Therapist: Let's start with this image. Can you describe what you see here?
    Image description: A red ball on a green lawn.
    Patient: Ba... ball.
    Therapist: Yes, that's right! It's a ball. Can you tell me what color it is?
    Patient: Re... red.
    Therapist: Excellent! It's a red ball. Let's move on to the next problem.

    Example conversation 2:
    Therapist: Look at this picture. What's in it?
    Image description: A cat sleeping on a sofa.
    Patient: Cat... sleep.
    Therapist: Yes, good job! The cat is sleeping. Where is the cat sleeping?
    Patient: So... sofa.
    Therapist: That's correct, the cat is sleeping on the sofa. Let's move on to the next problem.

    Example conversation 3:
    Therapist: Can you describe what this image shows?
    Image description: A blue car parked outside a house.
    Patient: Car... blue.
    Therapist: Right, it's a car and it's blue. Where is the car?
    Patient: House... outside.
    Therapist: Perfect, the car is parked outside a house. Let's move on to the next problem.

    Example conversation 4:
    Therapist: What do you see in this image?
    Image description: Three apples on a table.
    Patient: Ap... apples.
    Therapist: Yes, they are apples. How many apples are there?
    Patient: Three.
    Therapist: That's right, there are three apples. Let's move on to the next problem.

    Example conversation 5:
    Therapist: Tell me about this picture.
    Image description: A person riding a bicycle in a park.
    Patient: Bike... ride.
    Therapist: Yes, someone is riding a bike. Where are they?
    Patient: Park.
    Therapist: Great! They are riding a bike in the park. Let's move on to the next problem."""

exercise_2_examples = f"""Exercise Description: This exercise is focused on forming active sentences. Start with a simple sentence and then a more complex one.

    Example conversation 1:
    Therapist: Let's make a sentence with these words: 'dog', 'runs', 'park'.
    Patient: Dog... park.
    Therapist: Yes, the dog is in the park. What is the dog doing?
    Patient: Runs.
    Therapist: That's right! The dog runs in the park.

    Example conversation 2:
    Therapist: Now, form a sentence using 'book', 'reading', 'girl'.
    Patient: Girl... book.
    Therapist: Good start. What is the girl doing with the book?
    Patient: Reading.
    Therapist: Excellent! The girl is reading a book.

    Example conversation 3:
    Therapist: Create a sentence from 'man', 'fishing', 'river'.
    Patient: Man... river.
    Therapist: Yes, the man is at the river. What is he doing there?
    Patient: Fishing.
    Therapist: Perfect! The man is fishing in the river.

    Example conversation 4:
    Therapist: Let's use these words: 'children', 'playing', 'school'.
    Patient: Children... school.
    Therapist: That's right. What are the children doing at the school?
    Patient: Play.
    Therapist: Exactly! The children are playing at school.

    Example conversation 5:
    Therapist: Form a sentence with 'birds', 'flying', 'sky'.
    Patient: Birds... sky.
    Therapist: Yes, the birds are in the sky. What are they doing?
    Patient: Flying.
    Therapist: Correct! The birds are flying in the sky."""

exercise_3_examples = f"""Exercise Description: This exercise is focused on forming passive sentences, starting simple and then increasing complexity.

    Example conversation 1:
    Therapist: Turn this into a passive sentence: 'The chef cooks the meal.'
    Patient: Meal... cooked.
    Therapist: Yes, let's complete it. 'The meal is cooked by the chef.'

    Example conversation 2:
    Therapist: Now, make a passive sentence from: 'The teacher is teaching the students.'
    Patient: Students... taught.
    Therapist: Very good! 'The students are being taught by the teacher.'

    Example conversation 3:
    Therapist: Create a passive sentence using: 'The gardener waters the plants.'
    Patient: Plants... watered.
    Therapist: Right!"""

exercise_examples = {'Exercise 1': exercise_1_examples, 'Exercise 2': exercise_2_examples, 'Exercise 3': exercise_3_examples}
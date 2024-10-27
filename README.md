# MathGen
## Product Description
MathGen is a user-friendly calculator and educational assistant that leverages the ChatGPT API to generate video explanations and solutions to math equations.

## User interface
### Frontend
Upon visiting the website, users can navigate between three main pages: **Home**, **Generate**, and **About Us**. On the Generate page, users can input any math equation into a designated text box. Additional buttons on the screen allow users to insert special characters that may not be readily available on their keyboards.

When the user clicks the Generate button, the equation entered in the text box is converted to LaTeX (a widely-used mathematics typesetting system) on the frontend using [MathQuill](https://github.com/mathquill/mathquill).The program then transmits the LaTeX equation from the frontend to the backend, initiating the video content generation process.

## Video + Audio Generation
### Backend
Backend Process
- LaTeX Processing: The entered LaTeX is sent to the backend, where it serves as input for the ChatGPT API.

- API Interaction: The assistant processes the LaTeX to instruct the GPT API on the necessary operations. The API then generates:
  - An audio script outlining the explanation.
  - A sequence of LaTeX lines detailing the equations to be displayed on the screen.
 
- Audio Generation: The generated script is fed into a model to produce a high-quality audio file that narrates the solution process.

- Video Creation: Step-by-step "images," each showcasing different equations, are created and overlaid in sequential order. This process creates the illusion of a cohesive video presentation.

- Combination: An object called video_input consolidates the audio and corresponding images for each step. The program then synchronizes the display of images with the playback of audio, resulting in a fluid and natural-sounding educational video.

### Video Upload (S3)
The video generated by the backend is uploaded to Amazon Web Services (AWS) using the Command Line Interface [(CLI)](https://aws.amazon.com/cli/#:~:text=Create%20Free%20Account-,AWS%20Command%20Line%20Interface,and%20automate%20them%20through%20scripts.). This process involves the following steps:

- Upload to S3: The generated video file is securely uploaded to an [AWS S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html).

- Link Generation: Upon successful upload, the AWS CLI returns a link to the video stored in S3.

- Frontend Integration: This link is then passed back to the frontend, where the user interface displays the video, allowing users to view the generated content seamlessly.

# Why Create MathGen
## Advanced Personalized Learning:
MathGen aims to revolutionize the way students engage with mathematics by providing advanced personalized learning experiences. By leveraging AI technology, MathGen tailors educational content to meet the unique needs of each learner. Here are some key benefits of this approach:

- Individualized Learning Paths: MathGen allows a student to learn at their skill level and desired pace despite rigid curriculums, offering tailored explanations and solutions that align with their needs.

- Diverse Learning Materials: Through dynamic video explanations and audio narratives, MathGen caters to different learning styles - such as visual and auditory - ensuring that all users can find an approach that resonates with them.

- Engaging Content: By presenting mathematical concepts through videos, MathGen maintains user engagement and motivates learners to explore topics in greater depth.

- Accessibility: With an online platform that can be accessed anytime and anywhere, MathGen provides students with the resources they need to study at their convenience, fostering a self-directed learning environment.

By prioritizing personalized learning, MathGen empowers students to overcome challenges, build confidence in their mathematical abilities, and develop a lifelong love for learning.

# What's Next

As MathGen continues to develop, several potential features could enhance the learning experience:

- Concept Explainer: A feature that allows users to input questions about mathematical concepts, providing detailed explanations and examples.

- Practice Problem Generator: A tool that offers a variety of practice problems tailored to the user's skill level, accompanied by video solutions for guided learning.

- Adaptive Learning: Future versions may implement adaptive algorithms that analyze user performance and adjust content difficulty accordingly.

- Subject Expansion: Future iterations may broaden the scope to include subjects beyond mathematics, creating a comprehensive educational platform.

These developments aim to further empower users in their academic journeys, making MathGen an even more effective learning assistant.


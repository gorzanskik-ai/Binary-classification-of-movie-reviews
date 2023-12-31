import streamlit as st
from gpt import generate_review
from clean_transform import clean_transform
from classification import classify

st.title('Interactive movie reviews binary classificator')

st.markdown("# Generate movie review with chatgpt")

with st.form(key='form'):
    prompt = st.text_input('Describe what kind of movie review you want to be written.')
    st.text(f'(Default: Write me a professional movie review)')

    submit_button = st.form_submit_button(label='Generate review')

    if submit_button:
        with st.spinner('Generating review...'):
            output = generate_review(prompt)

            with open(r'data/output.txt', 'w') as file:
                file.write(str(output))
            file.close()

        # generation = st.text_input(label='Movie review generated by chat-gpt', value=output)
        st.text('Movie review generated by chat-gpt:')
        st.markdown(output)
        st.markdown("____")

    submit_generation = st.form_submit_button(label='Classify')
    if submit_generation:
        with st.spinner('Classifying...'):
            with open(r'data/output.txt', 'r') as file:
                review = file.read()
            file.close()
            output = clean_transform(review)
            output = classify(output)
        st.text('Result:')
        st.markdown(review)
        st.text(output)
        st.markdown("____")

st.markdown('# Write your own movie review')
with st.form(key='form2'):
    review = st.text_input('Write your own movie review')

    submit_button2 = st.form_submit_button(label='Classify')

    if submit_button2:
        with st.spinner('Classifying...'):
            output = review
            output = clean_transform(output)
            output = classify(output)
        st.text('Result:')
        st.text(output)

        st.markdown("____")

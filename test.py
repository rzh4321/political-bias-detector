import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

posts = [
    {"content": "Liberals want to increase taxes!", "label": "conservative"},
    {"content": "Conservatives lack compassion.", "label": "liberal"},
    {"content": "We need stronger borders and immigration reform.", "label": "conservative"},
    {"content": "Healthcare should be a universal right.", "label": "liberal"},
    {"content": "The free market will provide the best outcomes.", "label": "conservative"},
    {"content": "We must take immediate action on climate change.", "label": "liberal"},
    {"content": "Gun ownership is a fundamental right protected by the Second Amendment.", "label": "conservative"},
    {"content": "Education funding should be increased to provide equal opportunities for all.", "label": "liberal"},
    {"content": "Welfare programs discourage hard work and personal responsibility.", "label": "conservative"},
    {"content": "Women should have the right to make choices about their own bodies.", "label": "liberal"},
    {"content": "Taxes are too high and stifle economic growth.", "label": "conservative"},
    {"content": "We need to invest in renewable energy to create jobs and reduce dependence on fossil fuels.", "label": "liberal"},
    {"content": "The government should not interfere in the private sector.", "label": "conservative"},
    {"content": "Racial justice and police reform are necessary to address systemic inequality.", "label": "liberal"},
    {"content": "Traditional values should be upheld.", "label": "conservative"},
    {"content": "The minimum wage needs to be raised to a living wage.", "label": "liberal"},
    {"content": "School choice gives parents the freedom to choose their children's education path.", "label": "conservative"},
    {"content": "LGBTQ+ rights are human rights.", "label": "liberal"},
    {"content": "Government regulation often does more harm than good.", "label": "conservative"},
    {"content": "It's essential to provide a pathway to citizenship for undocumented immigrants.", "label": "liberal"},
    {"content": "Big government is inefficient and should be downsized.", "label": "conservative"},
    {"content": "The war on drugs has failed and we need to decriminalize drug use.", "label": "liberal"},
    {"content": "Entrepreneurship and small businesses are the backbone of our economy.", "label": "conservative"},
    {"content": "Prison reform is necessary to address the injustices of the criminal justice system.", "label": "liberal"},
    {"content": "The right to bear arms must not be infringed upon.", "label": "conservative"},
    {"content": "Public transportation should be expanded and made more efficient.", "label": "liberal"},
    {"content": "Foreign aid is a waste of taxpayer dollars.", "label": "conservative"},
    {"content": "Access to affordable housing is a critical issue that must be addressed.", "label": "liberal"},
    {"content": "Excessive government spending is leading us towards a fiscal crisis.", "label": "conservative"},
    {"content": "We should work towards a society where everyone has the freedom to marry whom they love.", "label": "liberal"},
    {"content": "Elizabeth Warren's wealth tax plan is the best way to reduce inequality.", "label": "liberal"},
    {"content": "Ron DeSantis' educational reforms protect parental rights.", "label": "conservative"},
    {"content": "Bernie Sanders' stance on free college tuition is essential for the future of our youth.", "label": "liberal"},
    {"content": "Ted Cruz's support for the oil industry is crucial for our national economy.", "label": "conservative"},
    {"content": "Alexandria Ocasio-Cortez's Green New Deal is a bold step towards tackling climate change.", "label": "liberal"},
    {"content": "Donald Trump's approach to foreign policy puts America's interests first.", "label": "conservative"},
    {"content": "Kamala Harris's plan to reform the criminal justice system addresses past injustices.", "label": "liberal"},
    {"content": "Mitch McConnell's stance on Supreme Court nominations has reshaped the judiciary.", "label": "conservative"},
    {"content": "Joe Biden's infrastructure bill is a necessary investment in America's future.", "label": "liberal"},
    {"content": "Nikki Haley's perspective on international relations emphasizes strong alliances.", "label": "conservative"},
    {"content": "Pete Buttigieg's focus on transportation innovation is paving the way for the future.", "label": "liberal"},
    {"content": "Mike Pence's conservative values reflect the heart of American tradition.", "label": "conservative"},
    {"content": "Ilhan Omar's advocacy for refugees is a reflection of America's humanitarian spirit.", "label": "liberal"},
    {"content": "Josh Hawley's critique of Big Tech is vital to protect privacy and free speech.", "label": "conservative"},
    {"content": "Gavin Newsom's policies on gun control are essential to reducing violence.", "label": "liberal"},
    {"content": "Rand Paul's defense of civil liberties is paramount to our constitutional rights.", "label": "conservative"},
    {"content": "Stacey Abrams' work on voting rights ensures our democracy remains strong.", "label": "liberal"},
    {"content": "Marco Rubio's foreign policy positions support democracy around the world.", "label": "conservative"},
    {"content": "Andrew Yang's ideas on universal basic income could revolutionize our economy.", "label": "liberal"},
    {"content": "Lindsey Graham's commitment to a strong military ensures our national security.", "label": "conservative"},
    {"content": "Donald Trump's policies on trade brought back American jobs.", "label": "conservative"},
    {"content": "Trump's rhetoric is divisive and not representative of our nation's values.", "label": "liberal"},
    {"content": "The Trump administration's tax cuts significantly boosted the economy.", "label": "conservative"},
    {"content": "Donald Trump's handling of the COVID-19 pandemic was inadequate and cost lives.", "label": "liberal"},
    {"content": "Trump's stance on China is necessary to protect intellectual property rights and national security.", "label": "conservative"},
    {"content": "The Trump administration's environmental policies rolled back decades of progress on climate action.", "label": "liberal"},
    {"content": "Donald Trump's appointment of conservative judges will positively impact the judiciary for generations.", "label": "conservative"},
    {"content": "Trump's immigration policies were inhumane and tarnished America's reputation as a land of opportunity.", "label": "liberal"},
    {"content": "The First Step Act, signed by Trump, made significant improvements to criminal justice reform.", "label": "conservative"},
    {"content": "Trump's frequent use of social media undermined the dignity of the presidency.", "label": "liberal"},
    {"content": "Under Trump, America regained its status and leverage in international politics.", "label": "conservative"},
    {"content": "The Trump administration's deregulation efforts hurt environmental protection efforts.", "label": "liberal"},
    {"content": "Trump's economic policies led to a historically low unemployment rate before the pandemic.", "label": "conservative"},
    {"content": "Donald Trump's false claims about election fraud undermined democracy.", "label": "liberal"},
    {"content": "The 'America First' policy of Trump's presidency was necessary for American prosperity.", "label": "conservative"},
    {"content": "The Trump administration's withdrawal from the Paris Agreement was a setback for global climate efforts.", "label": "liberal"}
    

]

test_posts = [
    {"content": "We should tax the rich a lot less. It's not fair.", "label": "conservative"},
    {"content": "We need to increase the minimum wage to support more families.", "label": "liberal"},
    {"content": "Woman should not have the right to an abortion. It is unethical.", "label": "conservative"},
    {"content": "We need to focus a lot more on renewable energy.", "label": "liberal"},
    {"content": "Donald trump is a good person", "label": "conservative"},
    {"content": "Hillary clinton would've made a good president", "label": "liberal"},
    {"content": "Joe Biden is too old to be a president. His ideas can't be trusted.", "label": "conservative"},


]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase all texts
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    # Reconstruct the text
    clean_text = ' '.join(lemmatized_words)
    return clean_text

# Clean each post
for post in posts:
    post['content'] = clean_text(post['content'])

# print(posts)

# `posts` is now a list of dictionaries with 'content' and 'label' as keys, like this:
# posts = [
#     {"content": "liberals want to increase taxes", "label": "conservative"},
#     {"content": "conservatives lack compassion", "label": "liberal"},
#     # ... (more cleaned posts)
# ]

# Extract the content and labels into separate lists
contents = [post['content'] for post in posts]
labels = [post['label'] for post in posts]

# Clean each post in the testing set using the clean_text function
for post in test_posts:
    post['content'] = clean_text(post['content'])

# Extract the content and labels into separate lists
test_contents = [post['content'] for post in test_posts]
test_labels = [post['label'] for post in test_posts]

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training set and transform the training set
X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=42)
tfidf_train_matrix = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set with the same vectorizer
tfidf_test_matrix = tfidf_vectorizer.transform(X_test)


# Now `tfidf_train_matrix` and `y_train` are ready to be used in the training of the ML model
# `tfidf_test_matrix` and `y_test` will be used for evaluating the model's performance

# Initialize the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(tfidf_train_matrix, y_train)

# Predict the labels for the test set
y_pred = clf.predict(tfidf_test_matrix)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{class_report}")
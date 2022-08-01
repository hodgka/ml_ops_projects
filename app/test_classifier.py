from server import NewsCategoryClassifier


model = NewsCategoryClassifier({
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        },
        "classes": [
                'Business',
                'Entertainment',
                'Health',
                'Music Feeds',
                'Sci/Tech',
                'Software and Developement',
                'Sports',
                'Toons',
            ]
    })

input = {
  "source": "BBC Technology",
  "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm",
  "title": "System gremlins resolved at HSBC",
  "description": "Computer glitches which led to chaos for HSBC customers on Monday are fixed, the High Street bank confirms."
}
print("+"*100)
print(model.predict_proba(input))

print("+"*100)
print("+"*100)
print(model.predict_label(input))
# scores = model.predict_proba(input)
print("+"*100)

inputs = [{
  "source": "BBC Technology",
  "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm",
  "title": "System gremlins resolved at HSBC",
  "description": "Computer glitches which led to chaos for HSBC customers on Monday are fixed, the High Street bank confirms."
},
{
  "source": "Yahoo World",
  "url": "http://us.rd.yahoo.com/dailynews/rss/world/*http://story.news.yahoo.com/news?tmpl=story2u=/nm/20050104/bs_nm/markets_stocks_us_europe_dc",
  "title": "Wall Street Set to Open Firmer (Reuters)",
  "description": "Reuters - Wall Street was set to start higher on\Tuesday to recoup some of the prior session's losses, though high-profile retailer Amazon.com  may come under\pressure after a broker downgrade."
},
{
  "source": "New York Times",
  "url": "",
  "title": "Weis chooses not to make pickoff",
  "description": "Bill Belichick won't have to worry about Charlie Weis raiding his coaching staff for Notre Dame. But we'll have to see whether new Miami Dolphins coach Nick Saban has an eye on any of his former assistants."
},
{
  "source": "Boston Globe",
  "url": "http://www.boston.com/business/articles/2005/01/04/mike_wallace_subpoenaed?rss_id=BostonGlobe--BusinessNews",
  "title": "Mike Wallace subpoenaed",
  "description": "Richard Scrushy once sat down to talk with 60 Minutes correspondent Mike Wallace about allegations that Scrushy started a huge fraud while chief executive of rehabilitation giant HealthSouth Corp. Now, Scrushy wants Wallace to do the talking."
},
{
  "source": "Reuters World",
  "url": "http://www.reuters.com/newsArticle.jhtml?type=worldNewsstoryID=7228962",
  "title": "Peru Arrests Siege Leader, to Storm Police Post",
  "description": "LIMA, Peru (Reuters) - Peruvian authorities arrested a former army major who led a three-day uprising in a southern  Andean town and will storm the police station where some of his  200 supporters remain unless they surrender soon, Prime  Minister Carlos Ferrero said on Tuesday."
},
{
  "source": "The Washington Post",
  "url": "http://www.washingtonpost.com/wp-dyn/articles/A46063-2005Jan3.html?nav=rss_sports",
  "title": "Ruffin Fills Key Role",
  "description": "With power forward Etan Thomas having missed the entire season, reserve forward Michael Ruffin has done well in taking his place."
}]


print("+"*100)
for input in inputs:
    print(input)
    print(f"{model.predict_label(input)} - {model.predict_proba(input)}")
    print()
    print()
    print()
    print("+"*100)


class_names = model.pipeline.classes_
# pred_proba = model.pipeline.predict_proba([document])
# pred_label_scores = {
#     class_names[i]: pred_proba[0][i]
#     for i in range(len(class_names))
# }
print(class_names)
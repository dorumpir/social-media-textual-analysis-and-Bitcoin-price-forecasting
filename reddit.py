import os
import pandas as pd
import datetime as dt

import praw
from pmaw import PushshiftAPI
# to use PMAW
api = PushshiftAPI(num_workers=32)
# to use PRAW
reddit = praw.Reddit(
    client_id = "",
    client_secret = "",
    username = "",
    password = "",
    user_agent = "my agent"
)

subreddits = ['btc', 'Bitcoin', 'CryptoCurrency', 'CryptoMarkets']
start_dt = dt.datetime(2020, 1, 1)
end_dt = dt.datetime(2022, 1, 1)
delta = dt.timedelta(days=5)
windows_limit = 500
# directory on which to store the data
basecorpus = './data/reddit/'

import time
def log_action(action):
    print(action)
    return


import pandas as pd
import datetime as dt


for subreddit in subreddits:
    dirpath = basecorpus
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    action = "\t[Subreddit] " + subreddit
    log_action(action)

    subredditdirpath = dirpath# + '/' + subreddit
    if os.path.exists(subredditdirpath):
        pass#continue
    else:
        os.makedirs(subredditdirpath)

    submissions_csv_path = subreddit + '-submissions.csv'
    from collections import defaultdict
    submissions_dict = defaultdict(list)
    # submissions_dict = {
    #     "id" : [],
    #     "url" : [],
    #     "title" : [],
    #     "score" : [],
    #     "num_comments": [],
    #     "created_utc" : [],
    #     "selftext" : [],
    # }

    beg, end = start_dt, end_dt
    start_time = time.time()

    while beg < end:
        # timestamps that define window of posts
        next_beg = beg + delta
        if next_beg.year != beg.year:
            next_beg = dt.datetime(next_beg.year, 1, 1)
        ts_after = int(beg.timestamp())
        ts_before = int(next_beg.timestamp())


        # use PSAW only to get id of submissions in time interval
        filter = ["id", "url", "title", "score", "num_comments", "created_utc", "selftext", 'upvote_ratio']
        for _ in range(10):
            try:
                gen = api.search_submissions(
                    after=ts_after,
                    before=ts_before,
                    filter=filter,
                    subreddit=subreddit,
                    # limit=windows_limit,
                    mem_safe=True,
                    safe_exit=True,
                )
                break
            except Exception as e:
                print(e)
                import time; time.sleep(5)
                api = PushshiftAPI(num_workers=32)
                continue


        # use PRAW to get actual info and traverse comment tree
        for submission_pmaw in gen:
            # use psaw here
            submission_id = submission_pmaw['id']
            # use praw from now on
            submission_praw = reddit.submission(id=submission_id)

            submissions_dict["id"].append(submission_praw.id)
            submissions_dict["url"].append(submission_praw.url)
            submissions_dict["title"].append(submission_praw.title)
            submissions_dict["score"].append(submission_praw.score)
            submissions_dict["num_comments"].append(submission_praw.num_comments)
            submissions_dict["created_utc"].append(submission_praw.created_utc)
            submissions_dict["selftext"].append(submission_praw.selftext)
            submissions_dict["upvote_ratio"].append(submission_praw.upvote_ratio)

            
            # no comments !!!
            submission_praw = submission_pmaw
            for key in filter:
                submissions_dict[key].append(submission_praw.get(key, None))
                
            submission_comments_csv_path = subreddit + '-submission_' + submission_praw["id"] + '-comments.csv'
            submission_comments_dict = {
                "comment_id" : [],
                "comment_parent_id" : [],
                "comment_body" : [],
                "comment_link_id" : [],
            }



            # extend the comment tree all the way
            # submission_praw.comments.replace_more(limit=None)
            # # for each comment in flattened comment tree
            # for comment in submission_praw.comments.list():
            #     submission_comments_dict["comment_id"].append(comment.id)
            #     submission_comments_dict["comment_parent_id"].append(comment.parent_id)
            #     submission_comments_dict["comment_body"].append(comment.body)
            #     submission_comments_dict["comment_link_id"].append(comment.link_id)
            #
            # # for each submission save separate csv comment file
            # pd.DataFrame(submission_comments_dict).to_csv(subredditdirpath + '/' + submission_comments_csv_path,
            #                                               index=False)

        action = "\t\t[Subreddit] " + subreddit + '\t' + str(beg) + ' ' + str(len(submissions_dict['id']))
        log_action(action)
        beg = next_beg

    # single csv file with all submissions
    pd.DataFrame(submissions_dict).to_csv(subredditdirpath + '/' + submissions_csv_path, index=False)

    action = f"\t\t[Info] Found submissions: {pd.DataFrame(submissions_dict).shape[0]}"
    log_action(action)

    action = f"\t\t[Info] Elapsed time: {time.time() - start_time: .2f}s"
    log_action(action)
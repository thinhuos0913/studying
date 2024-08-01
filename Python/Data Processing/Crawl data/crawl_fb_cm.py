import random
import pandas as pd
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys

# 1. Khai báo browser
browser = webdriver.Chrome(executable_path="./chromedriver.exe")
# print("Current session is {}".format(browser.session_id))
# 2. Mở URL của post
browser.get("https://www.facebook.com/Yugiohmod/?hc_ref=ARSfl1D1RCk2KBj9lL_zWPFAtLTo0fnHdBXiQvTuALXj7XkdqRLOLtsBmAiZVV2BI6o&fref=nf&__tn__=kC-R")

sleep(random.randint(5,10))
# browser.close()

# # 3. Lấy link hiện comment
showcomment_link = browser.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div[2]/div[2]/div/div[3]/div[2]/div/div[1]/div/div[2]/div/div[1]/div[2]/div/div/div[2]/div[2]/form/div/div[2]/div[1]/div/div[3]/span[1]/a")
showcomment_link.click()
sleep(random.randint(5,10))
# browser.quit()
# Get more comments:
morecomment_link = browser.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div[2]/div[2]/div/div[3]/div[2]/div/div[1]/div/div[2]/div/div[1]/div[2]/div/div/div[2]/div[2]/form/div/div[3]/div[2]/div/a/div/span")
morecomment_link.click()
sleep(random.randint(5,10))
# browser.quit()
# # 5. Tìm tất cả các comment và ghi ra màn hình (hoặc file)
# # -> lấy all thẻ div có thuộc tính aria-label='Bình luận'
# comment_list = browser.find_elements_by_xpath("//div[@aria-label='Bình luận']")
comment_list = browser.find_elements_by_xpath("//div[@aria-label='Bình luận']")
# print(comment_list)
# sleep(random.randint(5,10))
# browser.quit()

# # Lặp trong tất cả các comment và hiển thị nội dung comment ra màn hình
posters = []
comments = []
for comment in comment_list:
    # hiển thị tên người và nội dung, cách nhau bởi dấu :
    poster = comment.find_element_by_class_name("_6qw4")
    posters.append(poster.text)
    content = comment.find_element_by_class_name("_3l3x")
    comments.append(content.text)
    # print(poster.text)
    # print(content.text)
    print("*", poster.text,":", content.text)

# Save comments: if you're passing scalar values, you have to pass an index. So you can either not use scalar values for the columns
# -- e.g. use a list for Poster and Content [poster.text], [content.text]
df_dict = {'Poster': posters, 'Comment': comments}
df = pd.DataFrame(df_dict)
df.to_csv('yg_comment.csv')

sleep(random.randint(5,10))
browser.quit()
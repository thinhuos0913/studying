import random
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
# from selenium.common.exceptions import InvalidSessionIdException

# 1. Khai báo browser
browser = webdriver.Chrome(executable_path="./chromedriver.exe")
# print("Current session is {}".format(browser.session_id))
# 2. Mở URL của post
browser.get("https://www.facebook.com/groups/miaigroup/permalink/730028114435130/")

sleep(random.randint(5,10))
# browser.quit()

# # 3. Lấy link hiện comment
# showcomment_link = browser.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div[4]/div/div/div[1]/div/div[1]/div/div[2]")
# showcomment_link.click()
# sleep(random.randint(5,10))
# browser.quit()

# # 4. Lấy comment
showmore_link = browser.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div[4]/div/div/div[2]/div[2]")
showmore_link.click()
sleep(random.randint(5,10))
# browser.quit()
# sleep(random.randint(5,10))

# showmore_link.click()
# sleep(random.randint(5,10))

# # 5. Tìm tất cả các comment và ghi ra màn hình (hoặc file)
# # -> lấy all thẻ div có thuộc tính aria-label='Bình luận'
# comment_list = browser.find_elements_by_xpath("//div[@aria-label='Bình luận']")
comment_list = browser.find_elements_by_xpath("//div[@aria-label='Bình luận']")
# print(comment_list)
# sleep(random.randint(5,10))
# browser.quit()

# # Lặp trong tất cả các comment và hiển thị nội dung comment ra màn hình
# for comment in comment_list:
    # hiển thị tên người và nội dung, cách nhau bởi dấu :
    # poster = comment.find_element_by_class_name("_6qw4")
    # content = comment.find_element_by_class_name("_3l3x")
    # print(poster.text)
    # print(content.text)
    # print("*", poster.text,":", content.text)


# sleep(random.randint(5,10))

# # 6. Đóng browser
# browser.quit()
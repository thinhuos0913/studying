from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
# Initialize browser variable:
browser = webdriver.Chrome(executable_path='chromedriver.exe')
# Open browser:
browser.get('https://www.facebook.com/')
# Fill in user information: email, pass, login
txtUser = browser.find_element_by_id("email")
txtUser.send_keys("0988607818") # Fill user name

txtPass = browser.find_element_by_id("pass")
txtPass.send_keys("thomas1991")

txtPass.send_keys(Keys.ENTER)

# Stop browser 5s to check web load
sleep(5)
# Close browser
browser.close()
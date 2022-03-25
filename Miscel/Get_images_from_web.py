from selenium import webdriver


chrome_options = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images": 2}


chrome_options.add_experimental_option("prefs", prefs)

prefs = {"download.default_directory": "D:/Dataset/modifyed"}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get('https://data.csail.mit.edu/graphics/fivek/')

links = driver.find_elements_by_partial_link_text("TIFF16");

for link in links:
    link.click();

print('wololo')
# hrefs=[]
# for link in links:
#     hrefs.append(link.get_attribute("href"))
#
#
#
#
#
#
#
# driver.find_element_by_id("LinkNoticia").click()
#
# title = driver.find_element_by_id("cuDetalle_cuTitular_tituloNoticia")
# print(title.text)


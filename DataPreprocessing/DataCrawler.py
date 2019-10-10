"""
Crawler the crop price data from http://agmarknet.gov.in

author: Alexander Woodruff, Hangzhi Guo
"""
import selenium
from DataPreprocessing import CROPS, STATES
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options


# lists of scraped and unscraped pages after algorithm is run
scrapedPages = []
unscrapedPages = []

url = 'http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx'

# creates a webdriver using chrome and navigates to URL

chrome_options = Options()
chrome_options.add_argument("--headless")       # define headless

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)


# checks to see if element exists and returns appropriate boolean
def checkElementExistsByID(table):
    try:
        driver.find_element_by_id(table)
    except NoSuchElementException:
        return False
    return True


# checks to see if searching term is present in the dropdown values and returns proper boolean
def checkElementInDropdown(param, val):
    inputParam = driver.find_element_by_id(param)
    return val in inputParam.text


def process(years, months, states, crops):
    # list of all pages queried
    datesAndCrops = []

    # loops through all possible combinations of form inputs given parameters above
    for l in range(len(crops)):
        for i in range(len(years)):
            for j in range(len(months)):
                for k in range(len(states)):

                    datesAndCrops.append(years[i] + ' ' + months[j] + ' ' + states[k] + ' ' + crops[l])

                    # creates file names for each queried page regardless if it is used
                    saveFile = f"{months[j][:3]}_{years[i][2:]}_{states[k][:4]}_{crops[l][:4]}.csv"

                    # inputs the year into the form
                    inputYear = driver.find_element_by_id("cphBody_cboYear")
                    inputYear.send_keys(str(years[i]))

                    # inputs the month into the form
                    inputMonth = driver.find_element_by_id("cphBody_cboMonth")
                    inputMonth.send_keys(str(months[j]))

                    # wait stage for state dropdown to appear
                    wait = WebDriverWait(driver, 60)
                    wait.until(EC.presence_of_element_located((By.ID, 'cphBody_cboState')))

                    # finds dropdown for state and inputs valid state
                    inputState = driver.find_element_by_id("cphBody_cboState")
                    if checkElementInDropdown('cphBody_cboState', str(states[k])):
                        inputState.send_keys(str(states[k]))

                        # wait stage for commodity dropdown to appear
                        wait.until(EC.presence_of_element_located((By.ID, 'cphBody_cboCommodity')))

                        # finds dropdown for commodity and inputs valid commodity
                        inputCrop = driver.find_element_by_id('cphBody_cboCommodity')
                        if checkElementInDropdown('cphBody_cboCommodity', str(crops[l])):
                            inputCrop.send_keys(str(crops[l]))

                            # wait stage for submit button
                            wait.until(EC.presence_of_element_located((By.ID, 'cphBody_btnSubmit')))

                            # clicks submit button
                            driver.find_element_by_name("ctl00$cphBody$btnSubmit").click()

                            # wait stage for page to finish loading, not necessary but there as a precaution
                            time.sleep(3)

                            # checks to see if table is present on page refresh
                            if checkElementExistsByID('cphBody_gridRecords'):
                                tab_data = driver.find_element_by_id("cphBody_gridRecords")

                                # scrapes the rows from teh table
                                list_rows = [[cell.text for cell in row.find_elements_by_tag_name('td')]
                                             for row in tab_data.find_elements_by_tag_name('tr')]

                                # puts rows into a dataframe
                                df = pd.DataFrame(list_rows)

                                # sets the column names as they appear in the table
                                df.columns = ['Market', 'Arrival Date', 'Arrivals (Tonnes)', 'Variety',
                                              'Minimum Price(Rs./Quintal)',
                                              'Maximum Price(Rs./Quintal)', 'Modal Price(Rs./Quintal)']
                                df = df.iloc[1:]

                                # replaces white space in table with nan value
                                df = df.apply(lambda x: x.str.strip()).replace('', np.nan)

                                # nan value is then converted into the appropriate market name
                                for val in range(len(df['Market'])):
                                    if pd.isnull(df['Market'].values[val]):
                                        # Sets nan value to current market
                                        df['Market'].values[val] = market
                                    else:
                                        # Sets the market to the current value, indicates a new market
                                        market = df['Market'].values[val]

                                # scraped and cleaned data is saved to csv file in directory
                                df.to_csv("./Data/" + saveFile, index=False)
                                print('File ' + saveFile + ' scraped and saved')

                                # appends queried page information to scraped pages list
                                scrapedPages.append(years[i] + ' ' + months[j] + ' ' + states[k] + ' ' + crops[l])

                                # navigates the browser back to the form page to continue loop
                                driver.back()

                            else:
                                # no table is present (no data was reported)
                                print('No table for: ' + saveFile + '. No file saved')

                                # appends queried page information to unscraped pages list
                                unscrapedPages.append(years[i] + ' ' + months[j] + ' ' + states[k] + ' ' + crops[l])
                                driver.back()

                        else:
                            # the queried crop is not in the dropdown tab
                            print('No crop present for: ' + saveFile + '. No file saved')

                            # appends queried page information to unscraped pages list
                            unscrapedPages.append(years[i] + ' ' + months[j] + ' ' + states[k] + ' ' + crops[l])

                            # refreshes the page to continue the loop
                            driver.refresh()
                    else:
                        # the queried state is not in the dropdown tab
                        print('No state present for: ' + saveFile + '. No file saved')

                        # appends queried page information to unscraped pages list
                        unscrapedPages.append(years[i] + ' ' + months[j] + ' ' + states[k] + ' ' + crops[l])

                        # refreshes the page to continue the loop
                        driver.refresh()

    # creates a log file and saves the scraped pages list
    with open(f"{years[0]} Scraped Page Logs.txt", "w") as output:
        output.write(str(scrapedPages))

    # creates a log file and saves the unscraped pages list
    with open(f"{years[0]} Unscraped Page Logs.txt", "w") as output:
        output.write(str(unscrapedPages))

    # indicates that the program has completed
    print('Done')


if __name__ == '__main__':
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']

    process(years, months, STATES, CROPS)

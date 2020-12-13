'''
from twilio.rest import Client

account_sid = "ACbfe2ada64f41c41b0854e7b483735232"
auth_token = "086d3a23a1336240b36926abb3572983"
Spaces = 13
Cars = 1
client = Client(account_sid, auth_token)
content = "There are 14 slots in total, " + str(Cars) + " of them occupied, " + str(Spaces) + " of them available." + ""
message = client.messages.create(to="+8618290262011", from_="+16516614003", body=content)
'''

from tkinter import *
from twilio.rest import Client

root = Tk()
root.geometry("640x320")

def sendMessage():
    spaces = 14 - cars_num
    content = "Available parking slots are: " + str(empty_slots) + ". " + str(cars_num) + " of them occupied, " + str(spaces) + " of them available." + ""
    myLabel = Label(root, text=content)
    myLabel.pack()
    print(content)
    account_sid = "ACf56c120a6cd8977b5d936fbd72c11f87"
    auth_token = "fa1a9f2b1db7fb9b42eb164f631a7251"
    client = Client(account_sid, auth_token)
    message = client.messages.create(to="+14152037291", from_="、+14438430254", body=content)

'''
def send_text():
    account_sid = "ACbfe2ada64f41c41b0854e7b483735232"
    auth_token = "086d3a23a1336240b36926abb3572983"

    client = Client(account_sid, auth_token)
    content = "There are 14 slots in total, " + str(car_nums) + " of them occupied, " + str(spaces) + " of them available." + ""
    message = client.messages.create(to="+8618290262011", from_="+16516614003", body=content)
'''

with open('cas_infor.txt', 'r') as f:
    data = f.readlines()
f.close()

global empty_slots
empty_slots = []
global cars_num
cars_num = int(data[0][14])

for k in range(14):
    if int(data[0][k]) == 0:
        empty_slots.append(k)

myButton = Button(root, text="Send", command=sendMessage)
myButton.pack()

root.mainloop()
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
    account_sid = "xxxxxx"
    auth_token = "xxxxx"
    client = Client(account_sid, auth_token)
    message = client.messages.create(to="+xxxxxx", from_="、+xxxxxx", body=content)


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

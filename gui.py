#from chatterbot import ChatBot
#from chatterbot.trainers import ListTrainer
from tkinter import *
#bot = ChatBot("My Bot")

main = Tk()

main.geometry("500x650")

main.title("My Chat bot")
img = PhotoImage(file="bot1.png")

photoL = Label(main, image=img)

photoL.pack(pady=5)


def ask_from_bot():
    print("clicked")
    query = textF.get()
    # answer_from_bot = bot.get_response(query)
    # msgs.insert(END, "you : " + query)
    # print(type(answer_from_bot))
    # msgs.insert(END, "bot : " + str(answer_from_bot))
    # textF.delete(0, END)

frame = Frame(main)

sc = Scrollbar(frame)
msgs = Listbox(frame, width=80, height=20)

sc.pack(side=RIGHT, fill=Y)

msgs.pack(side=LEFT, fill=BOTH, pady=10)

frame.pack()

# creating text field

textF = Entry(main, font=("Verdana", 20))
textF.pack(fill=X, pady=10)

btn = Button(main, text="send", font=("Verdana", 20), command=ask_from_bot)
btn.pack()

main.mainloop()

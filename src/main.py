from time import sleep

word = "Hello World!"

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXZY !@#$%^&*()_+-={}|[]\:";'<>?,./'

iter_word = ""

for i in range(len(word)):

    for char in chars:
        word_i = iter_word  + char
        print(word_i)
        sleep(0.01)
        if char == word[i]:
            iter_word += char
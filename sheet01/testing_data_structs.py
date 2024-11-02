import shelve
 
rated_by = shelve.open("rated_by")
user_col = shelve.open("user_col")

for key in user_col:
    print(user_col[key].toarray())
    break 
rated_by.close()
user_col.close()
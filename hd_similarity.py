import numpy as np
import operator
import math

from hiper import buildIDModel, cosAngle, binarizeHV

def read_train_data(path):
    users = set()
    items = set()
    user_itemlist = {}
    item_userlist = {]

    for line in open(path,"r"):
        l = line.split("\t")
        user = int(l[0])
        item = int(l[1])
        users.add(user)
        items.add(item)

        if user_itemlist.has_key(user):
            user_itemlist[user].append(item)
        else:
            user_itemlist[user]= []
            user_itemslist[user].append(item)

        if item_userlist.has_key(item):
            user_itemlist[item].append(user)
        else:
            user_itemlist[item]= []
            user_itemslist[item].append(user)

    return users, items, user_itemlist, item_userlist

def read_test_data(path):
    users = set()
    items = set()
    user_itemlist = {}

    for line in open(path,"r"):
        l = line.split("\t")
        user = int(l[0])
        item = int(l[1])
        users.add(user)
        items.add(item)

        if user_items_dic.has_key(user):
            user_itemlist[user].append(item)
        else:
            user_itemlist[user] = []
            user_itemlist[user].append(item)

    return users, items, user_itemlist 

def calculate_similary(list_of_lists):


if __name__ == "__main__":
    data_path = "../data/ml-100k/"
    D = 10

    train_users, train_items, train_user_itemlist, train_item_userlist = read_train_data(data_path+"u1.base")
    test_users, test_items, test_user_itemlist = read_test_data(data_path+"u1.test")

    users = train_users.union(test_users)
    items = train_items.union(test_items)

    userMemory = {}
    userMemory, userModel =  buildIDModel(list(users), userMemory, D)

    productMemory = {}
    productMemory, productModel =  buildIDModel(list(items), productMemory, D)

    userItemMemory = []

    for u in list(train_users):
        itemlist = train_user_items_dic[u]
        addHV = np.zeros(D)

        for i in itemlist:
            addHV = addHV + productMemory[i]

        userItemMemory.append(addHV)

    test_user_scores = {]
    for u in list(test_users):
        for i in list(test_items):
            for j in item_userlist[i]:

















        











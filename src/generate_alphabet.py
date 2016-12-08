__author__ = 'juliusskye'

def main():
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    Zero26 = ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"

    # print "%s = (1%s)"%(alphabet[0], Zero26[:50])
    # for letter_index in range(1,len(alphabet)):
    #     print "%s = (%s,1%s)"%(alphabet[letter_index], Zero26[1:letter_index*2],Zero26[:50-letter_index*2])

    #### matix format###

    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    # Zero26 = "00000000000000000000000000"
    # print ''.join(alphabet)
    # for letter_index in range(len(alphabet)):
    #     # print Zero26[:letter_index]
    #     # print Zero26[:25-letter_index]
    #     print "%s1%s%s"%( Zero26[:letter_index],Zero26[:25-letter_index], alphabet[letter_index])


    ###### mnist format ####

    zero25 = "0," * 25
    #print "%s = (1,%s)" % (alphabet[0], zero25[:49])
    for idx in range (0, len (alphabet)):
        print "%s,1%s,%s" % (zero25[:idx*2-1], zero25[idx*2-1:-1],alphabet[idx])

if __name__ == "__main__":
    main()

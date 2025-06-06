cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C    IDEBLANK- removes all blanks and returns length
C
C    ireturn = IDEBLANK (string)
C
C       ireturn = length of the string without blanks
C        string = character string to be de-blanked
C
C-
      Integer Function ideblank(string)
      Character*256 string*(*),temp
      j=1
      temp=' '
      itemp=len(string)
      Do 10 i=1,itemp
      If (string(i:i).EQ.' ')Go To 10
      temp(j:j)=string(i:i)
      j=j+1
   10 Continue
      string=temp
      ideblank=j-1
      Return
      End

CS    USGS Function IIQUEST
C     Version: 1.0
C     Technical Contact: Richard W. Saltus
C     Release: not released
C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     Function IIQUEST
C
C     Program purpose:
C         This function asks a question which requires an integer as response.
C         If a default is available, the user may take it by simply pressing
C         return. If a non-integer response is given, the question is re-asked.
C         Use of the special string "//" allows the user to signal the main
C         program (usually used to flag the end of a list or desire to exit
C         a program section).
C
C     Instructions for use:
C         This function should be used to ask all questions requiring integral
C         response.
C
      INTEGER FUNCTION IIQUEST(QUEST,IVAL,FORM,MODE)
C-
C
C     Variables and parameters:
C
C    IIQUEST - Asks a question with an integer answer
C
C     ireturn = IIQUEST (quest,ival,form,mode)
C
C      ireturn =  -1 if '//' was given as response (user wants out)
C                  0 if no response (user took default)
C                  1 if user responded (returns response in aval)
C
C        quest =  Character string containing question to be asked
C                 (with no ? at the end, it is added by function)
C
C         ival =  integer variable to receive answer
C                 (used to pass default if one is available)
C
C         form =  Character string containing fortran format to be
C                 used to format the default contained in ival
C
C         mode =  Integer control parameter:
C
C            mode = 0, required response (no default allowed)
C            mode <>0, default allowed
C
C         Calls: ITLEN, GETANS
C^
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Character quest*(*),form*(*),rstr2*30
      Character*100 str,form2*10,ans*30
      iiquest=0
      iqlen=itlen(quest)
C
C     Format default for insertion into question
C
      Write (rstr2,form)ival
      irlen=ideblank(rstr2)
      If (mode.NE.0) Then
C
C        Construct question with default in brackets
C
         str=quest(1:iqlen)//' ['//rstr2(1:irlen)//']?'
         islen=iqlen+irlen+4
      Else
C
C        No default, add '?' to question
C
         str=quest(1:iqlen)//'?'
         islen=iqlen+1
      End If
C
C     Make fortran format for printing of question
C
      Write (form2,105)islen
  105 Format ('(x,a',i3,',$)')
C
C     Repeats to here if question is re-asked
C
   13 Continue
C
C     call getans to obtain answer
C
      Call getans(str,form2,ans)
C
C     If GETANS is not used, the following two statements will ask question
C
C      Write (6,form2)str
C      Read (5,110)ans
  110 Format (a30)
C
C     Check answer
C
      ialen=ideblank(ans)
      If (ialen.NE.0) Then
C
C        Got an answer
C
         If (ans.EQ.'//') Then
            iiquest=-1
         Else
C
C           Convert answer to an integer
C
            iiquest=1
            Read (ans(1:ialen),120,err=25)ival
  120       Format (i11)
         End If
      Else
C
C        Didn't get an answer, if no default ask question again
C
         If (mode.EQ.0)Go To 13
      End If
      Return
C
C     Couldn't decode, give error and ask question again
C
   25 Continue
      Write (6,130)
  130 Format (' Please answer again, I expect a number.')
      Go To 13
      End

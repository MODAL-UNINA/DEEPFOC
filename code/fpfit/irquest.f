CS    USGS Function IRQUEST
C     Version: 1.0
C     Technical Contact: Richard W. Saltus
C     Release: not released
C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     Function IRQUEST
C
C     Program purpose:
C         This function asks a question requiring a real number response.
C         It allows a default answer (if the user simply presses return).
C         The question is re-asked if a non-numeric answer is supplied.
C         The special character sequence "//" can be answered to indicate
C         a special condition.
C
C     Instructions for use:
C         This function should be called to ask all questions requiring decimal
C         numbers as answers.
C
      INTEGER FUNCTION IRQUEST(QUEST,RVAL,FORM,MODE)
C-
C
C     Variables and parameters:
C
C     ireturn = IRQUEST (quest,rval,form,mode)
C
C      ireturn =  -1 if '//' was given as response (user wants out)
C                  0 if no response (user took default)
C                  1 if user responded (returns response in aval)
C
C        quest =  Character string containing question to be asked
C                 (with no ? at the end, it is added by function)
C
C         rval =  Real variable to receive answer
C                 (used to pass default if one is available)
C
C         form =  Character string containing fortran format to be
C                 used to format the default contained in rval
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
      irquest=0
      iqlen=itlen(quest)
C
C     Format default for inclusion in question
C
      Write (rstr2,form)rval
      irlen=ideblank(rstr2)
      If (mode.NE.0) Then
C
C     If default allowed, insert it in question
C
         str=quest(1:iqlen)//' ['//rstr2(1:irlen)//']?'
         islen=iqlen+irlen+4
      Else
C
C        No default, just add '?' to question
C
         str=quest(1:iqlen)//'?'
         islen=iqlen+1
      End If
C
C     Construct format for printing the question
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
C      If getans is not used, the following statements will ask question
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
            irquest=-1
         Else
C
C           Convert answer to real number
C
            irquest=1
            Read (ans(1:ialen),120,err=25)rval
  120       Format (f20.0)
         End If
      Else
C
C        Didn't get an answer, if no default re-ask question
C
         If (mode.EQ.0)Go To 13
      End If
      Return
C
C     Couldn't convert the answer to a real number
C
   25 Continue
      Write (6,130)
  130 Format (' Please answer again, I''m expecting a number.')
      Go To 13
      End

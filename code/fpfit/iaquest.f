CS    USGS Function IAQUEST
C     Version: 1.0
C     Technical Contact: R. Saltus
C     Release: not released
C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     Function IAQUEST
C
C     Program purpose:
C         Asks a question requiring a character-string response.
C         Allows the use of a default answer and will optionally upshift the
C         response.
C
C     Instructions for use:
C         This subroutine should be called whenever a character-string response
C         is called for in a FORTRAN program.
C
      Integer Function iaquest(quest,aval,form,mode)
C-
C
C     Variables and parameters:
C     ireturn = IAQUEST (quest,aval,form,mode)
C
C      ireturn =  -1 if '//' was given as response (user wants out)
C                  0 if no response (user took default)
C                  1 if user responded (returns response in aval)
C
C        quest =  Character string containing question to be asked
C                 (with no ? at the end, it is added by function)
C
C         aval =  Character string to receive answer
C                 (used to pass default if one is available)
C
C         form =  Character string containing fortran format to be
C                 used to read the user response
C
C         mode =  Integer control parameter:
C
C              mode > 0, default allowed
C              mode = 0, no default allowed, DO NOT upshift response
C note: changed for UNIX, Vax version used to upshift on mode=0
C              mode < 0, default allowed, upshift response
C
C         Calls ITLEN, UPSHIFT, GETANS
C^
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Character quest*(*),form*(*),aval*(*)
      Character*100 str,form2*10,ans*80
      iaquest=0
      iqlen=itlen(quest)
      irlen=itlen(aval)
C
C     If response is to be upshifted, upshift default
C
      If (mode.LT.0)Call upshift(aval)
      If (irlen.EQ.0)irlen=1
C
C     Insert default answer into question
C
      If (mode.NE.0) Then
         str=quest(1:iqlen)//' ['//aval(1:irlen)//']?'
         islen=iqlen+irlen+4
      Else
         str=quest(1:iqlen)//'?'
         islen=iqlen+1
      End If
C
C     Construct fortran format for printing of question
C
      Write (form2,105)islen
  105 Format ('(x,a',i3,',$)')
C
C     Repeats to here if question is re-asked
C
   13 Continue
C
C     Call getans to obtain answer (allows multiple answers per line)
C
      Call getans(str,form2,ans)
C
C     If getans is not used, the following two statements ask question
C
C       Write (6,form2)str
C       Read (5,form)ans
C
C     Check answer
C
      ialen=31
      ialen=itlen(ans)
      If (ialen.NE.0) Then
C
C     got an answer
C
C     find first non-blank character
C
      do 5,ifirst=1,ialen
      if (ans(ifirst:ifirst).ne.' ') goto 6
   5  continue
   6  continue
      ans(1:ialen)=ans(ifirst:ialen)
         If (ans(ifirst:ialen).EQ.'//') Then
            iaquest=-1
         Else
            iaquest=1
C
C           Move answer into AVAL
C
            aval(1:len(aval))=ans(1:len(aval))
            If (mode.LT.0)Call upshift(aval)
         End If
      Else
C
C        Didn't get an answer, if no default ask question again
C
         If (mode.EQ.0)Go To 13
      End If
      Return
      End

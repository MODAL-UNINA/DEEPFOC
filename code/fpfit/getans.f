cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
CS    Subroutine GETANS
C     Version: 1.0 (Unix F77) 3/87
C     Technical Contact: Richard W. Saltus
C     Release: not released
C
C     Subroutine GETANS
C
C     Program purpose:
C         This subroutine is called by IAQUEST, IIQUEST, and IRQUEST to
C         ask questions and get responses. It allows for buffering of
C         input, answers to multiple questions on a single line, echoing
C         of program dialog, or input of answers from a file.
C
C     GETANS - Reads next answer from saved command line,
C              prompts for new command line if old command line empty.
C              Answers in the command line are delimited by semi-colons.
C
C       GETANS recognizes the following special commands:
C
C              <filename - takes subsequent input from the file 'filename'
C                          NOTE: read on fortran unit 16
C              >filename - records questions and responses to 'filename'
C                          NOTE: written to fortran unit 17
C              >*        - echos questions and answers at terminal
C              ><        - stops recording
C              ><*       - stops echoing
C              #stuff#   - skips stuff between pound signs
C
C     Instructions for use:
C         Intended for use by the IxQUEST subroutines only.
C
       SUBROUTINE GETANS(QUEST,QFORM,ANS)
C-
C
C     Variables and parameters:
C
C
C     Call GETANS (quest, qform, ans)
C
C          quest = character variable, question to be asked
C
C          qform = character variable, fortran format for question
C
C          ans   = character variable to receive answer
C
C     Called by IIQUEST, IAQUEST, IRQUEST
C
C     Calls IGETTOK
C
C     Commons used : /GETCOM/ nextc, ifile, isfile, command
C
C^
C
      Character*(*) quest,qform,ans
      Common /getcom/nextc,ifile,isfile,command
      Character*(80) command,tok,form*40
      Logical asked,typed
C
C     initialize token
C
    1 Continue
      tok=' '
      lentok=0
      ic=len(command)
      it=len(tok)
      asked=.false.
      typed=.false.
C
C     If nothing in command line, ask question
C
    5 Continue
      If ((nextc.EQ.0).OR.(nextc.GT.ic)) Then
         If (ifile.EQ.0) Then
C
C        ask question at terminal
C
            asked=.true.
            typed=.true.
            Write (6,qform)quest
            Read (5,100)command
  100       Format (a80)
         Else
C
C        read command from file
C
            asked=.true.
         Read (ifile,100,End =99,err=99)command
      End If
      nextc=1
      End If
C
C     get a token from the command line
C
   10 Continue
      lenread=igettok(command(nextc:ic),tok(lentok+1:it),ierr,itype)
      nextc=lenread+nextc
      If (ierr.GE.0) Then
C
C     got a token
C
         If (tok(lentok+1:lentok+1).EQ.';') Then
C
C     got end of token
C
            If (nextc.GE.ic)nextc=0
         Else If (tok(lentok+1:lentok+1).EQ.'#') Then
C
C     skip over stuff between #'s in command line
C
            iskip=index(command(nextc:ic),'#')
            If (iskip.EQ.0) Then
C
C     skip rest of record
C
               nextc=0
               Go To 5
            Else
C
C     skip past next #
C
               nextc=nextc+iskip
               Go To 5
            End If
         Else
C
C     didn't get a ';' or a '#' - append new tok to old tok (reduce multiple
C                                 blanks to a single blank between tokens)
C
            newlen=itlen(tok(lentok+1:it))
            If (lenread.GT.newlen) Then
               newlen=newlen+1
               Do 20 icnt=newlen+lentok+1,lentok+1,-1
               itemp=icnt+1
               tok(itemp:itemp)=tok(icnt:icnt)
   20          continue
               tok(lentok+1:lentok+1)=' '
            End If
            lentok=lentok+newlen
            If (ierr.EQ.0) Then
C
C     hit end of command line
C
               nextc=0
            End If
C
C     keep reading command line
C
            Goto 10
         End If
      Else
C
C      didn't get a token (hit end of line)
C
         nextc=0
         If ((lentok.EQ.0).AND.(.NOT.asked))Goto 5
      End If
C
C      ready to return answer
C
      ans=' '
      If (lentok.GT.0) Then
         If (tok(1:1).EQ.'<') Then
C
C     open file for command reading
C
            If (ifile.NE.0)Close (ifile)
            Open (16,file=tok(2:lentok),form='formatted',
     &      status='old',err=98)
            ifile=16
            nextc=0
            Go To 1
         Else If (tok(1:1).EQ.'>') Then
            If (tok(2:2).EQ.'*') Then
C
C     set echo flag for command echoing at terminal
C
               iecho=1
            Else If (tok(2:2).EQ.'<') Then
               If (tok(3:3).EQ.'*') Then
C
C      turn off command echoing
C
                  iecho=0
               Else
C
C      close recording file
C
                  Close (isfile)
                  isfile=0
                  Go To 1
               End If
            Else
C
C     Open recording file
C
               If (isfile.NE.0)Close (isfile)
               Open (17,file=tok(2:lentok),form='formatted',status=
     &         'unknown',err=97)
               isfile=17
               Go To 1
            End If
         Else
C
C     pass answer back and quit
C
            ans=tok(1:lentok)
         End If
      End If
      If (isfile.NE.0) Then
C
C     record question and answer in recording file
C
         If (lentok.LE.0)tok(1:1)=' '
         ir=max(1,lentok)
         iq=itlen(quest)
         Write (form,103)iq,ir
  103    Format ('(1x,a1,a',i2,',a1,a',i2,')')
         Write (isfile,form)'#',quest(1:iq),'#',tok(1:ir)
      End If
      If ((iecho.NE.0).AND.(.NOT.typed)) Then
C
C     Echo question and answer at terminal
C
         If (lentok.LE.0)tok(1:1)=' '
         ir=max(1,lentok)
         iq=itlen(quest)
         Write (form,104)iq,ir
  104    Format ('(1x,a',i2,',x,a',i2,')')
         Write (6,form)quest(1:iq),tok(1:ir)
      End If
      Return
C
C     error on recording file open
C
   97 Continue
      If (lentok.GE.2) Then
         Write (form,105)lentok-1
  105    Format ('(1x,a35,a',i2,')')
         Write (6,form)'Unable to open the recording file: ',
     &          tok(2:lentok)
      Else
         Write (6,106)
  106    Format (1x,'No name specified for recording file.')
      End If
      Go To 1
C
C     error on command file open
C
   98 Continue
      If (lentok.GE.2) Then
         Write (form,110)lentok-1
         Write (6,form)'Unable to open the command file: ',
     &   tok(2:lentok)
  110    Format ('(1x,a33,a',i2,')')
      Else
         Write (6,111)
  111    Format (1x,'No name specified for command file.')
      End If
      Go To 1
C
C     hit end of command file
C
   99 Continue
      Close (ifile)
      ifile=0
      Go To 5
      End

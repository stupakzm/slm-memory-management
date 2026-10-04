;;; asq.el --- Ask the local asq engine about Emacs  -*- lexical-binding: t; -*-

;;; Commentary:

;; M-x asq QUESTION asks the local engine, searching the Emacs manuals.
;; C-u M-x asq QUESTION drops the domain filter and searches everything.
;; Output streams into the *asq* buffer.

;;; Code:

(defgroup asq nil
  "Ask the local asq engine."
  :group 'tools)

(defcustom asq-program "asq"
  "The asq executable."
  :type 'string)

(defcustom asq-db "data/index/phase11.db"
  "Index passed to asq with --db (relative paths resolve against the repo)."
  :type 'string)

(defcustom asq-domain "emacs"
  "Domain passed to asq with --domain, unless searching everywhere."
  :type 'string)

(defun asq--sentinel (process _event)
  "Append PROCESS's exit status to its buffer when nonzero."
  (when (memq (process-status process) '(exit signal))
    (let ((status (process-exit-status process))
          (buf (process-buffer process)))
      (when (and (/= status 0) (buffer-live-p buf))
        (with-current-buffer buf
          (let ((inhibit-read-only t))
            (save-excursion
              (goto-char (point-max))
              (insert (format "\n[asq exited with status %d]\n" status)))))))))

;;;###autoload
(defun asq (question &optional everywhere)
  "Ask QUESTION about Emacs; with EVERYWHERE (prefix arg) search all domains."
  (interactive
   (list (read-string "asq: "
                      (and (use-region-p)
                           (buffer-substring-no-properties
                            (region-beginning) (region-end))))
         current-prefix-arg))
  (let ((buf (get-buffer-create "*asq*")))
    (when-let* ((old (get-buffer-process buf)))
      (delete-process old))
    (with-current-buffer buf
      (let ((inhibit-read-only t))
        (special-mode)
        (erase-buffer)
        (insert "Q: " question "\n\n")))
    (display-buffer buf)
    (make-process
     :name "asq"
     :buffer buf
     :command (append (list asq-program "--db" asq-db)
                      (unless everywhere (list "--domain" asq-domain))
                      (list question))
     :connection-type 'pipe
     :noquery t
     :filter (lambda (proc string)
               (when (buffer-live-p (process-buffer proc))
                 (with-current-buffer (process-buffer proc)
                   (let ((inhibit-read-only t))
                     (save-excursion
                       (goto-char (point-max))
                       (insert string))))))
     :sentinel #'asq--sentinel)))

(provide 'asq)
;;; asq.el ends here
